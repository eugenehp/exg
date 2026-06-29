//! Source space generation.
//!
//! Defines where the dipole sources live inside the head volume.
//! Two methods are provided:
//!
//! - **Icosahedron subdivision**: recursively subdivided icosahedron
//!   projected onto a sphere, then scaled to a given radius. This
//!   approximates a cortical surface with roughly uniform spacing.
//!
//! - **Regular grid**: axis-aligned 3-D grid inside a sphere, useful
//!   for volume source spaces.
//!
//! ## Example
//!
//! ```
//! use exg_source::source_space::{ico_source_space, grid_source_space};
//!
//! // ~162 sources on a sphere of radius 0.07 m (cortex)
//! let (pos, nn) = ico_source_space(2, 0.07, [0.0, 0.0, 0.04]);
//! assert_eq!(pos.nrows(), nn.nrows());
//!
//! // Volume grid with 0.01 m spacing inside a 0.08 m sphere
//! let (pos, nn) = grid_source_space(0.01, 0.08, [0.0, 0.0, 0.04]);
//! ```

use ndarray::Array2;
use std::collections::HashMap;

/// Generate a source space by subdividing an icosahedron.
///
/// # Arguments
///
/// * `n_subdivisions` — Number of recursive subdivisions (0 → 12 vertices,
///   1 → 42, 2 → 162, 3 → 642, 4 → 2562, 5 → 10242).
/// * `radius` — Radius of the sphere in metres (e.g., 0.07 for cortex).
/// * `center` — Centre of the sphere `[x, y, z]` in metres.
///
/// # Returns
///
/// `(positions, normals)` where both have shape `[n_sources, 3]`.
/// Normals point radially outward from the centre.
pub fn ico_source_space(
    n_subdivisions: usize,
    radius: f64,
    center: [f64; 3],
) -> (Array2<f64>, Array2<f64>) {
    let (verts, faces) = make_icosahedron();
    let (verts, _faces) = subdivide_ico(verts, faces, n_subdivisions);

    let n = verts.len();
    let mut positions = Array2::zeros((n, 3));
    let mut normals = Array2::zeros((n, 3));

    for (i, v) in verts.iter().enumerate() {
        let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        let nx = v[0] / len;
        let ny = v[1] / len;
        let nz = v[2] / len;

        positions[[i, 0]] = nx * radius + center[0];
        positions[[i, 1]] = ny * radius + center[1];
        positions[[i, 2]] = nz * radius + center[2];

        normals[[i, 0]] = nx;
        normals[[i, 1]] = ny;
        normals[[i, 2]] = nz;
    }

    (positions, normals)
}

/// Generate a volume source space on a regular 3-D grid.
///
/// # Arguments
///
/// * `spacing` — Grid spacing in metres (e.g., 0.01 for 1 cm).
/// * `radius`  — Only include points inside a sphere of this radius.
/// * `center`  — Centre of the sphere `[x, y, z]` in metres.
///
/// # Returns
///
/// `(positions, normals)` where both have shape `[n_sources, 3]`.
/// For a volume source space, normals are set to `[0, 0, 1]` (arbitrary).
pub fn grid_source_space(
    spacing: f64,
    radius: f64,
    center: [f64; 3],
) -> (Array2<f64>, Array2<f64>) {
    assert!(spacing > 0.0, "Spacing must be positive");
    assert!(radius > 0.0, "Radius must be positive");

    let n_half = (radius / spacing).ceil() as i32;
    let mut points = Vec::new();

    for ix in -n_half..=n_half {
        for iy in -n_half..=n_half {
            for iz in -n_half..=n_half {
                let x = ix as f64 * spacing;
                let y = iy as f64 * spacing;
                let z = iz as f64 * spacing;
                if x * x + y * y + z * z <= radius * radius {
                    points.push([x + center[0], y + center[1], z + center[2]]);
                }
            }
        }
    }

    let n = points.len();
    let mut positions = Array2::zeros((n, 3));
    let mut normals = Array2::zeros((n, 3));

    for (i, p) in points.iter().enumerate() {
        positions[[i, 0]] = p[0];
        positions[[i, 1]] = p[1];
        positions[[i, 2]] = p[2];
        // Volume sources: arbitrary upward normal
        normals[[i, 2]] = 1.0;
    }

    (positions, normals)
}

/// Expected vertex count for a given icosahedron subdivision level.
pub fn ico_n_vertices(n_subdivisions: usize) -> usize {
    // V = 10 * 4^n + 2
    10 * 4_usize.pow(n_subdivisions as u32) + 2
}

// ── Icosahedron geometry ───────────────────────────────────────────────────

/// Create the base icosahedron (12 vertices, 20 faces).
fn make_icosahedron() -> (Vec<[f64; 3]>, Vec<[usize; 3]>) {
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0; // golden ratio
    let a = 1.0;
    let b = 1.0 / phi;

    // 12 vertices of the icosahedron (normalised to unit sphere)
    #[rustfmt::skip]
    let raw_verts: Vec<[f64; 3]> = vec![
        [ 0.0,  b, -a], [ b,  a,  0.0], [-b,  a,  0.0],
        [ 0.0,  b,  a], [-a,  0.0,  b], [ 0.0, -b,  a],
        [ a,  0.0,  b], [ a,  0.0, -b], [ 0.0, -b, -a],
        [-a,  0.0, -b], [-b, -a,  0.0], [ b, -a,  0.0],
    ];

    let verts: Vec<[f64; 3]> = raw_verts
        .iter()
        .map(|v| {
            let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            [v[0] / len, v[1] / len, v[2] / len]
        })
        .collect();

    #[rustfmt::skip]
    let faces: Vec<[usize; 3]> = vec![
        [0, 1, 2], [3, 2, 1], [3, 4, 5], [3, 5, 6],
        [0, 7, 1], [0, 8, 7], [0, 9, 8], [0, 2, 9],
        [3, 1, 6], [3, 4, 2], [5, 4, 10], [5, 10, 11],
        [5, 11, 6], [7, 6, 11], [7, 11, 8], [8, 11, 10],
        [8, 10, 9], [9, 10, 4], [9, 4, 2], [6, 7, 1],
    ];

    (verts, faces)
}

/// Subdivide an icosahedral mesh `n` times.
///
/// Each subdivision splits every triangle into 4 by inserting midpoints
/// on each edge and projecting them onto the unit sphere.
fn subdivide_ico(
    mut verts: Vec<[f64; 3]>,
    mut faces: Vec<[usize; 3]>,
    n: usize,
) -> (Vec<[f64; 3]>, Vec<[usize; 3]>) {
    for _ in 0..n {
        let mut edge_midpoint: HashMap<(usize, usize), usize> = HashMap::new();
        let mut new_faces = Vec::with_capacity(faces.len() * 4);

        for face in &faces {
            let [a, b, c] = *face;
            let ab = get_or_insert_midpoint(a, b, &mut verts, &mut edge_midpoint);
            let bc = get_or_insert_midpoint(b, c, &mut verts, &mut edge_midpoint);
            let ca = get_or_insert_midpoint(c, a, &mut verts, &mut edge_midpoint);

            new_faces.push([a, ab, ca]);
            new_faces.push([b, bc, ab]);
            new_faces.push([c, ca, bc]);
            new_faces.push([ab, bc, ca]);
        }

        faces = new_faces;
    }

    (verts, faces)
}

/// Get or create the midpoint vertex between two vertices, projected
/// onto the unit sphere.
fn get_or_insert_midpoint(
    i: usize,
    j: usize,
    verts: &mut Vec<[f64; 3]>,
    cache: &mut HashMap<(usize, usize), usize>,
) -> usize {
    let key = if i < j { (i, j) } else { (j, i) };
    if let Some(&idx) = cache.get(&key) {
        return idx;
    }

    let a = verts[i];
    let b = verts[j];
    let mx = (a[0] + b[0]) / 2.0;
    let my = (a[1] + b[1]) / 2.0;
    let mz = (a[2] + b[2]) / 2.0;
    let len = (mx * mx + my * my + mz * mz).sqrt();

    let idx = verts.len();
    verts.push([mx / len, my / len, mz / len]);
    cache.insert(key, idx);
    idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ico_vertex_counts() {
        assert_eq!(ico_n_vertices(0), 12);
        assert_eq!(ico_n_vertices(1), 42);
        assert_eq!(ico_n_vertices(2), 162);
        assert_eq!(ico_n_vertices(3), 642);
        assert_eq!(ico_n_vertices(4), 2562);
    }

    #[test]
    fn test_ico_source_space_shape() {
        for subdiv in 0..=3 {
            let (pos, nn) = ico_source_space(subdiv, 0.07, [0.0, 0.0, 0.04]);
            let expected = ico_n_vertices(subdiv);
            assert_eq!(
                pos.nrows(),
                expected,
                "subdiv={subdiv}: expected {expected} vertices, got {}",
                pos.nrows()
            );
            assert_eq!(pos.ncols(), 3);
            assert_eq!(nn.dim(), pos.dim());
        }
    }

    #[test]
    fn test_ico_source_space_radius() {
        let radius = 0.07;
        let center = [0.0, 0.0, 0.04];
        let (pos, _) = ico_source_space(2, radius, center);

        for i in 0..pos.nrows() {
            let dx = pos[[i, 0]] - center[0];
            let dy = pos[[i, 1]] - center[1];
            let dz = pos[[i, 2]] - center[2];
            let r = (dx * dx + dy * dy + dz * dz).sqrt();
            approx::assert_abs_diff_eq!(r, radius, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_ico_normals_unit_length() {
        let (_, nn) = ico_source_space(2, 0.07, [0.0, 0.0, 0.0]);
        for i in 0..nn.nrows() {
            let len = (nn[[i, 0]].powi(2) + nn[[i, 1]].powi(2) + nn[[i, 2]].powi(2)).sqrt();
            approx::assert_abs_diff_eq!(len, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_grid_source_space_inside_sphere() {
        let radius = 0.05;
        let center = [0.0, 0.0, 0.03];
        let (pos, _) = grid_source_space(0.01, radius, center);

        assert!(pos.nrows() > 0, "Should have some sources");
        for i in 0..pos.nrows() {
            let dx = pos[[i, 0]] - center[0];
            let dy = pos[[i, 1]] - center[1];
            let dz = pos[[i, 2]] - center[2];
            let r = (dx * dx + dy * dy + dz * dz).sqrt();
            assert!(
                r <= radius + 1e-10,
                "Source at r={r} exceeds radius={radius}"
            );
        }
    }

    #[test]
    fn test_grid_source_space_includes_center() {
        let center = [0.01, 0.02, 0.03];
        let (pos, _) = grid_source_space(0.01, 0.05, center);

        // The center (or very close to it) should be included
        let mut min_dist = f64::MAX;
        for i in 0..pos.nrows() {
            let dx = pos[[i, 0]] - center[0];
            let dy = pos[[i, 1]] - center[1];
            let dz = pos[[i, 2]] - center[2];
            let r = (dx * dx + dy * dy + dz * dz).sqrt();
            if r < min_dist {
                min_dist = r;
            }
        }
        assert!(
            min_dist < 0.015,
            "Closest point to center is {min_dist} m away"
        );
    }

    #[test]
    fn test_grid_spacing_smaller_gives_more_sources() {
        let (p1, _) = grid_source_space(0.02, 0.05, [0.0, 0.0, 0.0]);
        let (p2, _) = grid_source_space(0.01, 0.05, [0.0, 0.0, 0.0]);
        assert!(
            p2.nrows() > p1.nrows(),
            "Finer grid should have more sources: {} vs {}",
            p2.nrows(),
            p1.nrows()
        );
    }
}
