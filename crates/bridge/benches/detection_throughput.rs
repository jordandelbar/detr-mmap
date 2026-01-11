use bridge::{Detection, DetectionReader, DetectionWriter};
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use std::fs;

/// Benchmark writing detections with full serialization
fn benchmark_detection_write(c: &mut Criterion) {
    let mut group = c.benchmark_group("detection_write");

    // Realistic detection counts
    let detection_counts = [
        (0, "no_detections"),
        (1, "single_detection"),
        (5, "few_detections"),
        (20, "many_detections"),
        (100, "crowded_scene"),
    ];

    for (i, (count, label)) in detection_counts.iter().enumerate() {
        let path = format!("/tmp/bridge_bench_detection_write_{}", i);
        let _ = fs::remove_file(&path);

        let mut writer = DetectionWriter::build_with_path(
            &path,
            1024 * 1024, // 1MB buffer
        )
        .unwrap();

        // Create realistic detection data
        let detections: Vec<Detection> = (0..*count)
            .map(|i| Detection {
                x1: (i * 10) as f32,
                y1: (i * 10) as f32,
                x2: (i * 10 + 50) as f32,
                y2: (i * 10 + 50) as f32,
                confidence: 0.85,
                class_id: 0, // person
            })
            .collect();

        group.bench_with_input(BenchmarkId::new("serialize_write", label), label, |b, _| {
            let mut frame_count = 0u64;
            b.iter(|| {
                writer
                    .write(
                        black_box(frame_count),
                        black_box(1234567890),
                        black_box(0),
                        black_box(&detections),
                    )
                    .unwrap();
                frame_count += 1;
            });
        });

        let _ = fs::remove_file(&path);
    }

    group.finish();
}

/// Benchmark reading detections with full deserialization
fn benchmark_detection_read(c: &mut Criterion) {
    let mut group = c.benchmark_group("detection_read");

    let detection_counts = [
        (0, "no_detections"),
        (1, "single_detection"),
        (5, "few_detections"),
        (20, "many_detections"),
        (100, "crowded_scene"),
    ];

    for (i, (count, label)) in detection_counts.iter().enumerate() {
        let path = format!("/tmp/bridge_bench_detection_read_{}", i);
        let _ = fs::remove_file(&path);

        // Write detections first
        let mut writer = DetectionWriter::build_with_path(&path, 1024 * 1024).unwrap();

        let detections: Vec<Detection> = (0..*count)
            .map(|i| Detection {
                x1: (i * 10) as f32,
                y1: (i * 10) as f32,
                x2: (i * 10 + 50) as f32,
                y2: (i * 10 + 50) as f32,
                confidence: 0.85,
                class_id: 0,
            })
            .collect();

        writer.write(1, 1234567890, 0, &detections).unwrap();

        // Create reader
        let reader = DetectionReader::with_path(&path).unwrap();

        group.bench_with_input(
            BenchmarkId::new("deserialize_read", label),
            label,
            |b, _| {
                b.iter(|| {
                    let result = reader.get_detections().unwrap();
                    black_box(result);
                });
            },
        );

        let _ = fs::remove_file(&path);
    }

    group.finish();
}

/// Benchmark full write-read roundtrip
fn benchmark_detection_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("detection_roundtrip");

    let detection_counts = [
        (0, "no_detections"),
        (1, "single_detection"),
        (5, "few_detections"),
        (20, "many_detections"),
        (100, "crowded_scene"),
    ];

    for (i, (count, label)) in detection_counts.iter().enumerate() {
        let path = format!("/tmp/bridge_bench_detection_roundtrip_{}", i);
        let _ = fs::remove_file(&path);

        let mut writer = DetectionWriter::build_with_path(&path, 1024 * 1024).unwrap();

        let mut reader = DetectionReader::with_path(&path).unwrap();

        let detections: Vec<Detection> = (0..*count)
            .map(|i| Detection {
                x1: (i * 10) as f32,
                y1: (i * 10) as f32,
                x2: (i * 10 + 50) as f32,
                y2: (i * 10 + 50) as f32,
                confidence: 0.85,
                class_id: 0,
            })
            .collect();

        group.bench_with_input(BenchmarkId::new("full_cycle", label), label, |b, _| {
            let mut frame_count = 0u64;
            b.iter(|| {
                // Write (serialize + publish)
                writer
                    .write(
                        black_box(frame_count),
                        black_box(1234567890),
                        black_box(0),
                        black_box(&detections),
                    )
                    .unwrap();

                // Read (detect + deserialize)
                let result = reader.get_detections().unwrap();
                if let Some(dets) = result {
                    black_box(dets.len());
                }

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
    benchmark_detection_write,
    benchmark_detection_read,
    benchmark_detection_roundtrip
);
criterion_main!(benches);
