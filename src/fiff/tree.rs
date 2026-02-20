//! FIF directory tree construction.
//!
//! Mirrors `mne/_fiff/tree.py` but uses owned Rust types throughout.
//!
//! The tree is built by scanning all tag headers sequentially and grouping
//! them into blocks delimited by `FIFF_BLOCK_START` / `FIFF_BLOCK_END` tags.
use std::io::{Read, Seek};
use anyhow::Result;

use super::constants::*;
use super::tag::{read_i32, read_tag_header, TagHeader};

// ── Node ─────────────────────────────────────────────────────────────────

/// One node in the FIF tree, analogous to MNE's `dict` node.
#[derive(Debug, Default, Clone)]
pub struct Node {
    /// Block kind (e.g. `FIFFB_MEAS`, `FIFFB_RAW_DATA`, …).
    /// 0 = root / unknown.
    pub block:    i32,
    /// All non-structural tag headers in this node (not including BLOCK_START/END).
    pub entries:  Vec<TagHeader>,
    /// Child nodes.
    pub children: Vec<Node>,
}

impl Node {
    /// Recursively find the first child node with the given block kind.
    pub fn find_block(&self, kind: i32) -> Option<&Node> {
        if self.block == kind {
            return Some(self);
        }
        for child in &self.children {
            if let Some(found) = child.find_block(kind) {
                return Some(found);
            }
        }
        None
    }

    /// Recursively collect all nodes with the given block kind.
    pub fn find_blocks(&self, kind: i32) -> Vec<&Node> {
        let mut out = Vec::new();
        self.collect_blocks(kind, &mut out);
        out
    }

    fn collect_blocks<'a>(&'a self, kind: i32, out: &mut Vec<&'a Node>) {
        if self.block == kind {
            out.push(self);
        }
        for child in &self.children {
            child.collect_blocks(kind, out);
        }
    }

    /// Find the first tag header with the given kind in this node's entries.
    /// Does NOT recurse into children.
    pub fn find_tag(&self, kind: i32) -> Option<&TagHeader> {
        self.entries.iter().find(|e| e.kind == kind)
    }
}

// ── Tree builder ─────────────────────────────────────────────────────────

/// Build the tag tree by walking the flat directory sequentially.
///
/// `directory` is the ordered list of all tag headers in the file.
/// This matches `mne._fiff.tree.make_dir_tree()`.
pub fn build_tree(directory: &[TagHeader]) -> Node {
    let mut stack: Vec<Node> = vec![Node::default()]; // root
    for &tag in directory {
        match tag.kind {
            FIFF_BLOCK_START => {
                // Push a new child node. Block kind is not resolved here
                // (no file access); use `read_tree` if you need resolved kinds.
                stack.push(Node::default());
            }
            FIFF_BLOCK_END => {
                let finished = stack.pop().unwrap_or_default();
                if let Some(parent) = stack.last_mut() {
                    parent.children.push(finished);
                }
            }
            _ => {
                if let Some(node) = stack.last_mut() {
                    node.entries.push(tag);
                }
            }
        }
    }
    // Anything left on the stack belongs to root.
    while stack.len() > 1 {
        let orphan = stack.pop().unwrap();
        stack[0].children.push(orphan);
    }
    stack.pop().unwrap_or_default()
}

/// Walk a flat directory and build the tree, resolving block kinds from the file.
pub fn read_tree<R: Read + Seek>(reader: &mut R, directory: &[TagHeader]) -> Result<Node> {
    let mut root = Node::default();
    build_tree_resolved(reader, directory, &mut root)?;
    Ok(root)
}

fn build_tree_resolved<R: Read + Seek>(
    reader: &mut R,
    directory: &[TagHeader],
    root: &mut Node,
) -> Result<()> {
    let mut stack: Vec<Node> = vec![Node {
        block:    0, // root
        entries:  Vec::new(),
        children: Vec::new(),
    }];

    for &tag in directory {
        match tag.kind {
            FIFF_BLOCK_START => {
                let block_kind = read_i32(reader, &tag).unwrap_or(0);
                stack.push(Node {
                    block:    block_kind,
                    entries:  Vec::new(),
                    children: Vec::new(),
                });
            }
            FIFF_BLOCK_END => {
                let finished = stack.pop().unwrap_or_default();
                if let Some(parent) = stack.last_mut() {
                    parent.children.push(finished);
                }
            }
            _ => {
                if let Some(node) = stack.last_mut() {
                    node.entries.push(tag);
                }
            }
        }
    }

    while stack.len() > 1 {
        let orphan = stack.pop().unwrap();
        if let Some(parent) = stack.last_mut() {
            parent.children.push(orphan);
        }
    }

    *root = stack.pop().unwrap_or_default();
    Ok(())
}

// ── Directory scanner ─────────────────────────────────────────────────────

/// Read every tag header by following the `next` pointer chain.
/// This is MNE's "slow path" — used when there is no pre-built directory,
/// or when we want to build a fresh one.
pub fn scan_directory<R: Read + Seek>(reader: &mut R) -> Result<Vec<TagHeader>> {
    let mut directory = Vec::new();
    let mut pos: Option<u64> = Some(0);
    while let Some(p) = pos {
        let tag = read_tag_header(reader, p)?;
        pos = tag.next_pos();
        directory.push(tag);
    }
    Ok(directory)
}

// ── Fast directory from embedded dir tag ─────────────────────────────────

/// Try to load the pre-built tag directory embedded at the end of the file.
///
/// MNE checks `FIFF_DIR_POINTER` (tag kind 101) early in the file; if its
/// payload is > 0 it points to a `FIFFT_DIR_ENTRY_STRUCT` tag containing
/// all headers.  Returns `None` if missing or corrupt.
pub fn try_load_directory<R: Read + Seek>(reader: &mut R) -> Result<Option<Vec<TagHeader>>> {
    // First tag must be FIFF_FILE_ID.
    let id_tag = read_tag_header(reader, 0)?;
    if id_tag.kind != FIFF_FILE_ID {
        return Ok(None);
    }
    // Second tag must be FIFF_DIR_POINTER.
    let next = match id_tag.next_pos() {
        Some(p) => p,
        None => return Ok(None),
    };
    let dir_ptr_tag = read_tag_header(reader, next)?;
    if dir_ptr_tag.kind != FIFF_DIR_POINTER {
        return Ok(None);
    }
    let dirpos = read_i32(reader, &dir_ptr_tag)? as i64;
    if dirpos <= 0 {
        return Ok(None);
    }
    let dir_tag = read_tag_header(reader, dirpos as u64)?;
    if dir_tag.ftype != super::constants::FIFFT_DIR_ENTRY_STRUCT {
        return Ok(None);
    }
    let entries = super::tag::read_directory(reader, &dir_tag)?;
    Ok(Some(entries))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_flat_dir(spec: &[(i32, u32, i32, i32)]) -> Vec<TagHeader> {
        spec.iter()
            .enumerate()
            .map(|(i, &(kind, ftype, size, next))| TagHeader {
                kind, ftype, size, next, pos: (i as u64) * 100,
            })
            .collect()
    }

    #[test]
    fn flat_directory_no_blocks() {
        // No BLOCK_START/END → everything in root entries
        let dir = make_flat_dir(&[
            (FIFF_NCHAN, FIFFT_INT, 4, 0),
            (FIFF_SFREQ, FIFFT_FLOAT, 4, 0),
        ]);
        let root = build_tree(&dir);
        assert_eq!(root.entries.len(), 2);
        assert!(root.children.is_empty());
    }

    #[test]
    fn single_block() {
        let dir = make_flat_dir(&[
            (FIFF_BLOCK_START, FIFFT_INT, 4, 0), // FIFFB_MEAS
            (FIFF_NCHAN,       FIFFT_INT, 4, 0),
            (FIFF_BLOCK_END,   FIFFT_INT, 4, 0),
        ]);
        let root = build_tree(&dir);
        assert_eq!(root.children.len(), 1);
        assert_eq!(root.children[0].entries.len(), 1);
    }

    #[test]
    fn nested_blocks() {
        let dir = make_flat_dir(&[
            (FIFF_BLOCK_START, FIFFT_INT, 4, 0), // outer
            (FIFF_BLOCK_START, FIFFT_INT, 4, 0), // inner
            (FIFF_NCHAN,       FIFFT_INT, 4, 0),
            (FIFF_BLOCK_END,   FIFFT_INT, 4, 0), // close inner
            (FIFF_SFREQ,       FIFFT_FLOAT, 4, 0),
            (FIFF_BLOCK_END,   FIFFT_INT, 4, 0), // close outer
        ]);
        let root = build_tree(&dir);
        assert_eq!(root.children.len(), 1);           // outer
        let outer = &root.children[0];
        assert_eq!(outer.children.len(), 1);          // inner
        assert_eq!(outer.entries.len(), 1);           // SFREQ
        assert_eq!(outer.children[0].entries.len(), 1); // NCHAN
    }

    #[test]
    fn find_block_depth_first() {
        let dir = make_flat_dir(&[
            (FIFF_BLOCK_START, FIFFT_INT, 4, 0),
            (FIFF_BLOCK_START, FIFFT_INT, 4, 0),
            (FIFF_NCHAN,       FIFFT_INT, 4, 0),
            (FIFF_BLOCK_END,   FIFFT_INT, 4, 0),
            (FIFF_BLOCK_END,   FIFFT_INT, 4, 0),
        ]);
        let root = build_tree(&dir);
        // block kinds not set (no file reads in pure-flat test), but
        // find_block(0) should find root.
        assert!(root.find_block(0).is_some());
    }
}
