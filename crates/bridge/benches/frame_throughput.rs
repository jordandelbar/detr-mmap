use bridge::{FrameReader, FrameWriter};
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use std::fs;

/// Benchmark writing frames with full serialization
fn benchmark_frame_write(c: &mut Criterion) {
    let mut group = c.benchmark_group("frame_write");

    // Realistic frame sizes
    let sizes = [
        (640, 480, "VGA"),       // 640x480x3 = ~900KB
        (1280, 720, "HD"),       // 1280x720x3 = ~2.7MB
        (1920, 1080, "Full HD"), // 1920x1080x3 = ~6MB
    ];

    for (i, (width, height, label)) in sizes.iter().enumerate() {
        let path = format!("/tmp/bridge_bench_frame_write_{}", i);
        // Remove file if it exists to force create_and_init
        let _ = fs::remove_file(&path);

        // Allocate extra space for FlatBuffers overhead (header + metadata)
        let buffer_size = (width * height * 3 + 8192) as usize;

        let mut writer = FrameWriter::build_with_path(&path, buffer_size).unwrap();

        // Create realistic frame data
        let pixel_data = vec![128u8; (width * height * 3) as usize];

        group.bench_with_input(BenchmarkId::new("serialize_write", label), label, |b, _| {
            let mut frame_count = 0u64;
            b.iter(|| {
                writer
                    .write_frame(
                        black_box(0),
                        black_box(&pixel_data),
                        black_box(frame_count),
                        black_box(*width),
                        black_box(*height),
                        black_box(None),
                    )
                    .unwrap();
                frame_count += 1;
            });
        });

        // Cleanup
        let _ = fs::remove_file(&path);
    }

    group.finish();
}

/// Benchmark reading frames with full deserialization
fn benchmark_frame_read(c: &mut Criterion) {
    let mut group = c.benchmark_group("frame_read");

    let sizes = [
        (640, 480, "VGA"),
        (1280, 720, "HD"),
        (1920, 1080, "Full HD"),
    ];

    for (i, (width, height, label)) in sizes.iter().enumerate() {
        let path = format!("/tmp/bridge_bench_frame_read_{}", i);
        let _ = fs::remove_file(&path);

        let buffer_size = (width * height * 3 + 8192) as usize;

        // Write a frame first
        {
            let mut writer = FrameWriter::build_with_path(&path, buffer_size).unwrap();
            let pixel_data = vec![128u8; (width * height * 3) as usize];
            writer
                .write_frame(0, &pixel_data, 1, *width, *height, None)
                .unwrap();
        }

        let reader = FrameReader::with_path(&path).unwrap();

        group.bench_with_input(
            BenchmarkId::new("deserialize_read", label),
            label,
            |b, _| {
                b.iter(|| {
                    let frame = reader.get_frame().unwrap();
                    black_box(frame);
                });
            },
        );

        let _ = fs::remove_file(&path);
    }

    group.finish();
}

/// Benchmark full write-read roundtrip (realistic use case)
fn benchmark_frame_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("frame_roundtrip");

    let sizes = [
        (640, 480, "VGA"),
        (1280, 720, "HD"),
        (1920, 1080, "Full HD"),
    ];

    for (i, (width, height, label)) in sizes.iter().enumerate() {
        let path = format!("/tmp/bridge_bench_frame_roundtrip_{}", i);
        let _ = fs::remove_file(&path);

        let buffer_size = (width * height * 3 + 8192) as usize;

        // Initialize file
        {
            let mut writer = FrameWriter::build_with_path(&path, buffer_size).unwrap();
            writer
                .write_frame(
                    0,
                    &vec![0u8; (width * height * 3) as usize],
                    0,
                    *width,
                    *height,
                    None,
                )
                .unwrap();
        }

        let mut reader = FrameReader::with_path(&path).unwrap();
        let mut writer = FrameWriter::build_with_path(&path, buffer_size).unwrap();

        let pixel_data = vec![128u8; (width * height * 3) as usize];

        group.bench_with_input(BenchmarkId::new("full_cycle", label), label, |b, _| {
            let mut frame_count = 0u64;
            b.iter(|| {
                writer
                    .write_frame(
                        black_box(0),
                        black_box(&pixel_data),
                        black_box(frame_count),
                        black_box(*width),
                        black_box(*height),
                        black_box(None),
                    )
                    .unwrap();

                let frame = reader.get_frame().unwrap().unwrap();
                black_box(frame.frame_number());
                black_box(frame.pixels());

                reader.mark_read();
                frame_count += 1;
            });
        });

        let _ = fs::remove_file(&path);
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_frame_write,
    benchmark_frame_read,
    benchmark_frame_roundtrip
);
criterion_main!(benches);
