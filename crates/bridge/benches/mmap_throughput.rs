use bridge::{FrameWriter, MmapReader};
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use tempfile::NamedTempFile;

fn benchmark_mmap_write(c: &mut Criterion) {
    let mut group = c.benchmark_group("mmap_write");

    let sizes = [
        (1024, "1KB"),            // Small metadata
        (10 * 1024, "10KB"),      // Compressed frame
        (100 * 1024, "100KB"),    // Small image
        (1024 * 1024, "1MB"),     // VGA raw frame (640x480x3)
        (6 * 1024 * 1024, "6MB"), // Full HD raw frame (1920x1080x3)
    ];

    for (size, label) in sizes.iter() {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        // Create writer with enough space (add 8 bytes for header)
        let mut writer = FrameWriter::new(path, size + 8).unwrap();
        let data = vec![0u8; *size];

        group.bench_with_input(BenchmarkId::new("write", label), size, |b, _| {
            b.iter(|| {
                writer.write(black_box(&data)).unwrap();
            });
        });
    }

    group.finish();
}

fn benchmark_mmap_read(c: &mut Criterion) {
    let mut group = c.benchmark_group("mmap_read");

    let sizes = [
        (1024, "1KB"),
        (10 * 1024, "10KB"),
        (100 * 1024, "100KB"),
        (1024 * 1024, "1MB"),
        (6 * 1024 * 1024, "6MB"),
    ];

    for (size, label) in sizes.iter() {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        let mut writer = FrameWriter::new(path, size + 8).unwrap();
        let data = vec![42u8; *size];
        writer.write(&data).unwrap();

        let mut reader = MmapReader::new(path).unwrap();

        group.bench_with_input(BenchmarkId::new("read", label), size, |b, _| {
            b.iter(|| {
                let buffer = reader.buffer();
                black_box(buffer);
                reader.mark_read();
            });
        });
    }

    group.finish();
}

fn benchmark_write_read_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("roundtrip");

    let sizes = [
        (100 * 1024, "100KB"),    // Small compressed frame
        (1024 * 1024, "1MB"),     // VGA raw
        (6 * 1024 * 1024, "6MB"), // Full HD raw
    ];

    for (size, label) in sizes.iter() {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        let mut writer = FrameWriter::new(path, size + 8).unwrap();
        let mut reader = MmapReader::new(path).unwrap();
        let data = vec![128u8; *size];

        group.bench_with_input(BenchmarkId::new("write_read_cycle", label), size, |b, _| {
            b.iter(|| {
                // Write frame
                writer.write(black_box(&data)).unwrap();

                // Read frame
                assert!(reader.has_new_data());
                let buffer = reader.buffer();
                black_box(buffer);
                reader.mark_read();
            });
        });
    }

    group.finish();
}

fn benchmark_sequence_check(c: &mut Criterion) {
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path();

    let mut writer = FrameWriter::new(path, 1024).unwrap();
    let reader = MmapReader::new(path).unwrap();
    let data = vec![0u8; 1000];

    // Write some data
    writer.write(&data).unwrap();

    c.bench_function("sequence_check", |b| {
        b.iter(|| {
            let has_new = reader.has_new_data();
            black_box(has_new);
        });
    });
}

fn benchmark_atomic_sequence_read(c: &mut Criterion) {
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path();

    let mut writer = FrameWriter::new(path, 1024).unwrap();
    let reader = MmapReader::new(path).unwrap();
    let data = vec![0u8; 1000];

    writer.write(&data).unwrap();

    c.bench_function("current_sequence", |b| {
        b.iter(|| {
            let seq = reader.current_sequence();
            black_box(seq);
        });
    });
}

criterion_group!(
    benches,
    benchmark_mmap_write,
    benchmark_mmap_read,
    benchmark_write_read_roundtrip,
    benchmark_sequence_check,
    benchmark_atomic_sequence_read
);
criterion_main!(benches);
