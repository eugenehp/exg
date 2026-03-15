/// Shared helpers for test vector loading.
use ndarray::{Array, IxDyn};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub fn vectors_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("vectors")
}

#[allow(unused)]
/// Load F32 tensors only (for DSP tests).
pub fn load_vectors(name: &str) -> HashMap<String, Array<f32, IxDyn>> {
    let path = vectors_dir().join(format!("{name}.safetensors"));
    let bytes = std::fs::read(&path)
        .unwrap_or_else(|_| panic!("test vector not found: {}", path.display()));

    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + n]).unwrap();
    let data_start = 8 + n;

    let mut out = HashMap::new();
    for (key, val) in header.as_object().unwrap() {
        if key == "__metadata__" { continue; }
        let dtype = val["dtype"].as_str().unwrap();
        if dtype != "F32" { continue; }
        let offsets = val["data_offsets"].as_array().unwrap();
        let s = offsets[0].as_u64().unwrap() as usize;
        let e = offsets[1].as_u64().unwrap() as usize;
        let raw = &bytes[data_start + s..data_start + e];
        let floats: Vec<f32> = raw.chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let shape: Vec<usize> = val["shape"].as_array().unwrap()
            .iter().map(|v| v.as_u64().unwrap() as usize).collect();
        let arr = Array::from_shape_vec(IxDyn(&shape), floats).unwrap();
        out.insert(key.clone(), arr);
    }
    out
}

#[allow(unused)]
/// Load all numeric tensors converted to f64.
/// Handles F32, F64, I32, I64.
pub fn load_vectors_f64(name: &str) -> HashMap<String, Array<f64, IxDyn>> {
    let path = vectors_dir().join(format!("{name}.safetensors"));
    let bytes = std::fs::read(&path)
        .unwrap_or_else(|_| panic!("test vector not found: {}", path.display()));

    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + n]).unwrap();
    let data_start = 8 + n;

    let mut out = HashMap::new();
    for (key, val) in header.as_object().unwrap() {
        if key == "__metadata__" { continue; }
        let dtype = val["dtype"].as_str().unwrap();
        let offsets = val["data_offsets"].as_array().unwrap();
        let s = offsets[0].as_u64().unwrap() as usize;
        let e = offsets[1].as_u64().unwrap() as usize;
        let raw = &bytes[data_start + s..data_start + e];
        let shape: Vec<usize> = val["shape"].as_array().unwrap()
            .iter().map(|v| v.as_u64().unwrap() as usize).collect();

        let vals: Vec<f64> = match dtype {
            "F32" => raw.chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()) as f64)
                .collect(),
            "F64" => raw.chunks_exact(8)
                .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
                .collect(),
            "I32" => raw.chunks_exact(4)
                .map(|b| i32::from_le_bytes(b.try_into().unwrap()) as f64)
                .collect(),
            "I64" => raw.chunks_exact(8)
                .map(|b| i64::from_le_bytes(b.try_into().unwrap()) as f64)
                .collect(),
            "U8"  => raw.iter().map(|&b| b as f64).collect(),
            _ => continue,
        };

        let arr = Array::from_shape_vec(IxDyn(&shape), vals).unwrap();
        out.insert(key.clone(), arr);
    }
    out
}

#[allow(unused)]
/// Maximum absolute difference between two arrays.
pub fn max_abs_diff(a: &Array<f32, IxDyn>, b: &Array<f32, IxDyn>) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0_f32, f32::max)
}

#[allow(unused)]
/// Standard deviation of an array.
pub fn array_std(a: &Array<f32, IxDyn>) -> f32 {
    let n = a.len() as f32;
    let mean: f32 = a.iter().sum::<f32>() / n;
    let var: f32 = a.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / n;
    var.sqrt()
}
