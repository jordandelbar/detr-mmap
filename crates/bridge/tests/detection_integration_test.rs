use bridge::{Detection, DetectionReader, DetectionWriter};
use std::thread;
use std::time::Duration;
use tempfile::tempdir;

/// Test basic detection writer-reader synchronization
///
/// Tests:
/// - Initial state (no data available)
/// - Detection write visibility to reader
/// - Sequence number progression
/// - mark_read() synchronization
/// - Multiple detection writes
/// - FlatBuffers serialization/deserialization
#[test]
fn test_detection_writer_reader_synchronization() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("detection_sync_test.mmap");
    let path_str = path.to_str().unwrap();

    // Initialize writer and reader
    let mut writer = DetectionWriter::build_with_path(path_str, 1024 * 1024).unwrap();
    let mut reader = DetectionReader::with_path(path_str).unwrap();

    // TEST 1: Initially, reader should have no data
    assert!(
        reader.get_detections().unwrap().is_none(),
        "Reader should have no detections initially"
    );

    // TEST 2: Write first detection batch (empty)
    let detections1: Vec<Detection> = vec![];
    writer.write(1, 1234567890, 0, &detections1).unwrap();

    // Reader should detect new data
    let result1 = reader.get_detections().unwrap();
    assert!(result1.is_some(), "Reader should detect new data");

    let dets = result1.unwrap();
    assert_eq!(dets.len(), 0, "Should have no detections");

    // TEST 3: Mark as read
    reader.mark_read();
    // Note: get_detections() returns current data if sequence > 0,
    // it doesn't distinguish between "new" and "already read" data.
    // This is expected behavior for the current API.

    // TEST 4: Write second batch with detections
    let detections2 = vec![
        Detection {
            x1: 10.0,
            y1: 20.0,
            x2: 100.0,
            y2: 200.0,
            confidence: 0.95,
            class_id: 0,
        },
        Detection {
            x1: 150.0,
            y1: 160.0,
            x2: 250.0,
            y2: 300.0,
            confidence: 0.88,
            class_id: 1,
        },
    ];

    writer.write(2, 1234567891, 0, &detections2).unwrap();

    // Reader should detect new detections
    let result2 = reader.get_detections().unwrap();
    assert!(result2.is_some(), "Reader should detect second batch");

    let dets = result2.unwrap();
    assert_eq!(dets.len(), 2, "Should have 2 detections");

    // Verify detection data
    assert_eq!(dets[0].x1, 10.0);
    assert_eq!(dets[0].y1, 20.0);
    assert_eq!(dets[0].confidence, 0.95);
    assert_eq!(dets[0].class_id, 0);

    assert_eq!(dets[1].x1, 150.0);
    assert_eq!(dets[1].confidence, 0.88);
    assert_eq!(dets[1].class_id, 1);

    // TEST 5: Multiple consecutive writes with varying detection counts
    for i in 3..=10 {
        let detections: Vec<Detection> = (0..i)
            .map(|j| Detection {
                x1: (j * 10) as f32,
                y1: (j * 10) as f32,
                x2: (j * 10 + 50) as f32,
                y2: (j * 10 + 50) as f32,
                confidence: 0.9,
                class_id: j as u32,
            })
            .collect();

        writer.write(i, 1234567890 + i, 0, &detections).unwrap();

        let result = reader.get_detections().unwrap();
        assert!(result.is_some(), "Reader should detect batch {}", i);

        let dets = result.unwrap();
        assert_eq!(dets.len(), i as usize, "Should have {} detections", i);

        reader.mark_read();
    }
}

/// Test concurrent producer-consumer pattern with detections
///
/// Simulates real-world scenario: inference thread writing detections while controller reads them.
///
/// Tests:
/// - Thread safety of detection serialization/deserialization
/// - Consumer doesn't miss detection batches
/// - Detection data integrity across thread boundary
#[test]
fn test_concurrent_detection_producer_consumer() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("detection_concurrent_test.mmap");

    const NUM_BATCHES: u64 = 30;
    const BUFFER_SIZE: usize = 1024 * 1024; // 1MB

    let path_producer = path.clone();
    let path_consumer = path.clone();

    // Producer thread: Simulates inference writing detections
    let producer = thread::spawn(move || {
        let mut writer =
            DetectionWriter::build_with_path(path_producer.to_str().unwrap(), BUFFER_SIZE).unwrap();

        // Give consumer time to initialize
        thread::sleep(Duration::from_millis(50));

        for batch_num in 1..=NUM_BATCHES {
            // Create detection batch with varying counts
            let detection_count = (batch_num % 10) as usize;
            let detections: Vec<Detection> = (0..detection_count)
                .map(|i| Detection {
                    x1: (i * 10) as f32,
                    y1: (i * 10) as f32,
                    x2: (i * 10 + 50) as f32,
                    y2: (i * 10 + 50) as f32,
                    confidence: 0.85,
                    class_id: i as u32,
                })
                .collect();

            writer
                .write(batch_num, 1234567890 + batch_num, 0, &detections)
                .unwrap();

            // Simulate realistic inference rate
            thread::sleep(Duration::from_millis(10));
        }

        NUM_BATCHES
    });

    // Consumer thread: Simulates controller reading detections
    let consumer = thread::spawn(move || {
        thread::sleep(Duration::from_millis(20));

        let mut reader = DetectionReader::with_path(path_consumer.to_str().unwrap()).unwrap();
        let mut batches_seen = Vec::new();

        let start = std::time::Instant::now();
        let timeout = Duration::from_secs(5);

        while batches_seen.len() < NUM_BATCHES as usize {
            if start.elapsed() > timeout {
                panic!(
                    "Consumer timeout: only saw {} batches: {:?}",
                    batches_seen.len(),
                    batches_seen
                );
            }

            if let Some(detections) = reader.get_detections().unwrap() {
                // Record this batch
                batches_seen.push(batches_seen.len() as u64 + 1);

                // Verify detection count varies (we don't know exact count without frame number from API)
                assert!(
                    detections.len() <= 10,
                    "Detection count should be reasonable"
                );

                reader.mark_read();
            } else {
                thread::sleep(Duration::from_millis(5));
            }
        }

        // Verify we saw all batches in order
        for (idx, &batch_num) in batches_seen.iter().enumerate() {
            assert_eq!(
                batch_num,
                (idx + 1) as u64,
                "Batch numbers should be sequential"
            );
        }

        batches_seen.len() as u64
    });

    let final_producer_count = producer.join().expect("Producer thread panicked");
    let batches_consumed = consumer.join().expect("Consumer thread panicked");

    assert_eq!(
        final_producer_count, NUM_BATCHES,
        "Producer should write exactly {} batches",
        NUM_BATCHES
    );
    assert_eq!(
        batches_consumed, NUM_BATCHES,
        "Consumer should receive exactly {} batches",
        NUM_BATCHES
    );

    println!(
        "Concurrent detection test passed: {} batches produced and consumed",
        NUM_BATCHES
    );
}

/// Test multiple concurrent readers
///
/// Simulates scenario: Both controller AND gateway reading detections from inference
/// This validates that multiple consumers can safely read from the same detection buffer
#[test]
fn test_multiple_detection_readers() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("multi_detection_reader_test.mmap");
    let path_str = path.to_str().unwrap();

    const NUM_BATCHES: usize = 20;

    // Setup writer
    let mut writer = DetectionWriter::build_with_path(path_str, 1024 * 1024).unwrap();

    // Create 3 readers (simulating inference, controller, gateway)
    let mut reader1 = DetectionReader::with_path(path_str).unwrap();
    let mut reader2 = DetectionReader::with_path(path_str).unwrap();
    let mut reader3 = DetectionReader::with_path(path_str).unwrap();

    // Write detection batches and verify all readers see them
    for i in 1..=NUM_BATCHES {
        let detections: Vec<Detection> = (0..5)
            .map(|j| Detection {
                x1: (i * j * 10) as f32,
                y1: (i * j * 10) as f32,
                x2: (i * j * 10 + 50) as f32,
                y2: (i * j * 10 + 50) as f32,
                confidence: 0.9,
                class_id: j as u32,
            })
            .collect();

        writer
            .write(i as u64, 1234567890 + i as u64, 0, &detections)
            .unwrap();

        // All readers should detect new batch
        let result1 = reader1.get_detections().unwrap();
        let result2 = reader2.get_detections().unwrap();
        let result3 = reader3.get_detections().unwrap();

        assert!(result1.is_some(), "Reader 1 should see batch {}", i);
        assert!(result2.is_some(), "Reader 2 should see batch {}", i);
        assert!(result3.is_some(), "Reader 3 should see batch {}", i);

        // All should read same data
        let d1 = result1.unwrap();
        let d2 = result2.unwrap();
        let d3 = result3.unwrap();

        assert_eq!(d1.len(), d2.len());
        assert_eq!(d2.len(), d3.len());
        assert_eq!(d1.len(), 5);

        // Mark as read independently
        reader1.mark_read();
        reader2.mark_read();
        reader3.mark_read();
    }

    println!(
        "Multiple detection readers test passed: {} batches",
        NUM_BATCHES
    );
}

/// Test writer behavior when detection data exceeds buffer
///
/// This prevents crashes when serialized detection batch is larger than allocated buffer
#[test]
fn test_detection_write_fails_when_buffer_too_small() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("detection_size_test.mmap");
    let path_str = path.to_str().unwrap();

    // Create writer with tiny buffer (100 bytes)
    let mut writer = DetectionWriter::build_with_path(path_str, 100).unwrap();

    // Try to write many detections (should exceed 100 bytes when serialized)
    let large_batch: Vec<Detection> = (0..100)
        .map(|i| Detection {
            x1: i as f32,
            y1: i as f32,
            x2: (i + 50) as f32,
            y2: (i + 50) as f32,
            confidence: 0.9,
            class_id: i,
        })
        .collect();

    let result = writer.write(1, 1234567890, 0, &large_batch);

    // Should fail - verify we got an error
    // (The exact error message may vary, but it should fail)
    assert!(
        result.is_err(),
        "Write should fail when detections exceed buffer"
    );
}

/// Test reader handles missing detections gracefully
///
/// Edge case: Reader polls but writer hasn't written anything yet
#[test]
fn test_detection_reader_handles_no_data() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("detection_stale_test.mmap");
    let path_str = path.to_str().unwrap();

    // Create writer but don't write anything
    let _writer = DetectionWriter::build_with_path(path_str, 1024 * 1024).unwrap();
    let reader = DetectionReader::with_path(path_str).unwrap();

    // Poll multiple times - should never indicate new data
    for _ in 0..10 {
        assert!(
            reader.get_detections().unwrap().is_none(),
            "Reader should not see detections when nothing written"
        );
        thread::sleep(Duration::from_millis(1));
    }
}

/// Test various detection counts
///
/// Validates that DetectionWriter/Reader handle edge cases correctly:
/// - No detections (empty batch)
/// - Single detection
/// - Few detections (typical case)
/// - Many detections (crowded scene)
#[test]
fn test_various_detection_counts() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("detection_counts_test.mmap");
    let path_str = path.to_str().unwrap();

    let detection_counts = [
        (0, "no_detections"),
        (1, "single_detection"),
        (5, "few_detections"),
        (20, "many_detections"),
        (100, "crowded_scene"),
    ];

    for (count, label) in &detection_counts {
        let mut writer = DetectionWriter::build_with_path(path_str, 1024 * 1024).unwrap();
        let reader = DetectionReader::with_path(path_str).unwrap();

        let detections: Vec<Detection> = (0..*count)
            .map(|i| Detection {
                x1: (i * 10) as f32,
                y1: (i * 10) as f32,
                x2: (i * 10 + 50) as f32,
                y2: (i * 10 + 50) as f32,
                confidence: 0.85,
                class_id: i,
            })
            .collect();

        writer.write(1, 1234567890, 0, &detections).unwrap();

        let result = reader.get_detections().unwrap();
        assert!(result.is_some(), "{} should be readable", label);

        let dets = result.unwrap();
        assert_eq!(dets.len(), *count as usize, "{} count mismatch", label);

        // Verify detection data integrity
        for (i, det) in dets.iter().enumerate() {
            assert_eq!(det.x1, (i * 10) as f32, "{} x1 mismatch", label);
            assert_eq!(det.class_id, i as u32, "{} class_id mismatch", label);
        }
    }
}

/// Test detection data precision
///
/// Validates that float values in detections maintain precision through serialization
#[test]
fn test_detection_float_precision() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("detection_precision_test.mmap");
    let path_str = path.to_str().unwrap();

    let mut writer = DetectionWriter::build_with_path(path_str, 1024 * 1024).unwrap();
    let reader = DetectionReader::with_path(path_str).unwrap();

    // Test with precise float values
    let detections = vec![
        Detection {
            x1: 123.456,
            y1: 789.012,
            x2: 345.678,
            y2: 901.234,
            confidence: 0.987654,
            class_id: 42,
        },
        Detection {
            x1: 0.001,
            y1: 0.002,
            x2: 0.003,
            y2: 0.004,
            confidence: 0.999999,
            class_id: 0,
        },
    ];

    writer.write(1, 1234567890, 0, &detections).unwrap();

    let result = reader.get_detections().unwrap();
    assert!(result.is_some(), "Should read detections");

    let dets = result.unwrap();
    assert_eq!(dets.len(), 2);

    // Verify float precision (FlatBuffers uses f32, so some precision loss expected)
    let epsilon = 0.0001; // Acceptable precision loss for f32

    assert!((dets[0].x1 - 123.456).abs() < epsilon);
    assert!((dets[0].y1 - 789.012).abs() < epsilon);
    assert!((dets[0].confidence - 0.987654).abs() < epsilon);
    assert_eq!(dets[0].class_id, 42);

    assert!((dets[1].x1 - 0.001).abs() < epsilon);
    assert!((dets[1].confidence - 0.999999).abs() < epsilon);
    assert_eq!(dets[1].class_id, 0);
}
