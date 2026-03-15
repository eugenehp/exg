use std::hint::black_box;
use criterion::{criterion_group, criterion_main, Criterion};
use exg::fiff::raw::open_raw;
use std::path::Path;

const FIF: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/data/sample1_raw.fif");

fn bench_open_raw(c: &mut Criterion) {
    if !Path::new(FIF).exists() {
        return;
    }
    c.bench_function("open_raw (header + tree)", |b| {
        b.iter(|| {
            let raw = open_raw(black_box(FIF)).unwrap();
            black_box(raw.info.n_chan)
        })
    });
}

fn bench_read_all_data(c: &mut Criterion) {
    if !Path::new(FIF).exists() {
        return;
    }
    let raw = open_raw(FIF).unwrap();
    c.bench_function("read_all_data [12×3840 f64]", |b| {
        b.iter(|| {
            let data = raw.read_all_data().unwrap();
            black_box(data.shape()[1])
        })
    });
}

fn bench_read_slice_1s(c: &mut Criterion) {
    if !Path::new(FIF).exists() {
        return;
    }
    let raw = open_raw(FIF).unwrap();
    c.bench_function("read_slice 256 samples (1 s)", |b| {
        b.iter(|| {
            let data = raw.read_slice(black_box(0), black_box(256)).unwrap();
            black_box(data[[0, 0]])
        })
    });
}

criterion_group!(benches, bench_open_raw, bench_read_all_data, bench_read_slice_1s);
criterion_main!(benches);
