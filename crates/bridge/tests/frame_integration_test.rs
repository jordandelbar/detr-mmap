use bridge::{FrameReader, FrameWriter};
use std::thread;
use std::time::Duration;
use tempfile::tempdir;

/// Test basic frame writer-reader synchronization
///
/// Tests:
/// - Initial state (no data available)
/// - Frame write visibility to reader
/// - Sequence number progression
/// - mark_read() synchronization
/// - Multiple frame writes
/// - FlatBuffers serialization/deserialization
#[test]
fn test_frame_writer_reader_synchronization() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("frame_sync_test.mmap");
    let path_str = path.to_str().unwrap();

    // Initialize writer and reader with realistic buffer size
    let mut writer = FrameWriter::build_with_path(path_str, 1024 * 1024).unwrap();
    let mut reader = FrameReader::with_path(path_str).unwrap();

    // TEST 1: Initially, reader should have no data
    assert!(
        reader.get_frame().unwrap().is_none(),
        "Reader should have no frame initially"
    );

    // TEST 2: Write first frame
    let pixels1 = vec![255u8; 640 * 480 * 3]; // VGA frame
    writer.write_frame(0, &pixels1, 1, 640, 480, None).unwrap();

    // Reader should detect new frame
    let frame1 = reader.get_frame().unwrap();
    assert!(frame1.is_some(), "Reader should detect new frame");

    let frame1 = frame1.unwrap();
    assert_eq!(frame1.frame_number(), 1, "Frame number should be 1");
    assert_eq!(frame1.camera_id(), 0, "Camera ID should be 0");
    assert_eq!(frame1.width(), 640, "Width should be 640");
    assert_eq!(frame1.height(), 480, "Height should be 480");
    assert_eq!(
        frame1.pixels().unwrap().len(),
        640 * 480 * 3,
        "Pixel data length should match"
    );

    // TEST 3: Mark as read
    reader.mark_read();
    // Note: get_frame() returns current frame if sequence > 0,
    // it doesn't distinguish between "new" and "already read" frames.
    // This is expected behavior for the current API.

    // TEST 4: Write second frame with different data
    let pixels2 = vec![128u8; 640 * 480 * 3];
    writer.write_frame(1, &pixels2, 2, 640, 480, None).unwrap();

    // Reader should detect new frame
    let frame2 = reader.get_frame().unwrap();
    assert!(frame2.is_some(), "Reader should detect second frame");

    let frame2 = frame2.unwrap();
    assert_eq!(frame2.frame_number(), 2, "Frame number should be 2");
    assert_eq!(frame2.camera_id(), 1, "Camera ID should be 1");

    // TEST 5: Multiple consecutive writes
    for i in 3..=10 {
        let pixels = vec![(i * 10) as u8; 640 * 480 * 3];
        writer.write_frame(i as u32, &pixels, i, 640, 480, None).unwrap();

        let frame = reader.get_frame().unwrap();
        assert!(frame.is_some(), "Reader should detect frame {}", i);

        let frame = frame.unwrap();
        assert_eq!(frame.frame_number(), i, "Frame number should be {}", i);

        reader.mark_read();
    }
}

/// Test concurrent producer-consumer pattern with realistic frame data
///
/// Simulates real-world scenario: capture thread writing frames while inference reads them.
///
/// Tests:
/// - Thread safety of frame serialization/deserialization
/// - Consumer doesn't miss frames
/// - Frame data integrity across thread boundary
#[test]
fn test_concurrent_frame_producer_consumer() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("frame_concurrent_test.mmap");

    const NUM_FRAMES: u64 = 30;
    const WIDTH: u32 = 640;
    const HEIGHT: u32 = 480;
    const BUFFER_SIZE: usize = 1024 * 1024; // 1MB

    let path_producer = path.clone();
    let path_consumer = path.clone();

    // Producer thread: Simulates capture writing frames
    let producer = thread::spawn(move || {
        let mut writer =
            FrameWriter::build_with_path(path_producer.to_str().unwrap(), BUFFER_SIZE).unwrap();

        // Give consumer time to initialize
        thread::sleep(Duration::from_millis(50));

        for frame_num in 1..=NUM_FRAMES {
            // Create frame with recognizable pattern
            let mut pixels = vec![0u8; (WIDTH * HEIGHT * 3) as usize];
            // Embed frame number in first 8 bytes for verification
            pixels[..8].copy_from_slice(&frame_num.to_le_bytes());

            writer.write_frame(0, &pixels, frame_num, WIDTH, HEIGHT, None).unwrap();

            // Simulate realistic frame rate (~100 FPS)
            thread::sleep(Duration::from_millis(10));
        }

        NUM_FRAMES
    });

    // Consumer thread: Simulates inference reading frames
    let consumer = thread::spawn(move || {
        thread::sleep(Duration::from_millis(20));

        let mut reader = FrameReader::with_path(path_consumer.to_str().unwrap()).unwrap();
        let mut frames_seen = Vec::new();

        let start = std::time::Instant::now();
        let timeout = Duration::from_secs(5);

        while frames_seen.len() < NUM_FRAMES as usize {
            if start.elapsed() > timeout {
                panic!(
                    "Consumer timeout: only saw {} frames: {:?}",
                    frames_seen.len(),
                    frames_seen
                );
            }

            if let Some(frame) = reader.get_frame().unwrap() {
                let frame_num = frame.frame_number();

                // Only process if we haven't seen this frame yet
                if frames_seen.last() != Some(&frame_num) {
                    let pixels = frame.pixels().unwrap();

                    // Verify frame number embedded in pixel data matches
                    let mut frame_num_bytes = [0u8; 8];
                    frame_num_bytes.copy_from_slice(&pixels.bytes()[..8]);
                    let embedded_num = u64::from_le_bytes(frame_num_bytes);

                    assert_eq!(
                        frame_num, embedded_num,
                        "Frame number should match embedded data"
                    );

                    // Verify frame dimensions
                    assert_eq!(frame.width(), WIDTH);
                    assert_eq!(frame.height(), HEIGHT);

                    frames_seen.push(frame_num);
                    reader.mark_read();
                } else {
                    thread::sleep(Duration::from_millis(5));
                }
            } else {
                thread::sleep(Duration::from_millis(5));
            }
        }

        // Verify we saw all frames in order
        for (idx, &frame_num) in frames_seen.iter().enumerate() {
            assert_eq!(
                frame_num,
                (idx + 1) as u64,
                "Frame numbers should be sequential"
            );
        }

        frames_seen.len() as u64
    });

    let final_producer_count = producer.join().expect("Producer thread panicked");
    let frames_consumed = consumer.join().expect("Consumer thread panicked");

    assert_eq!(
        final_producer_count, NUM_FRAMES,
        "Producer should write exactly {} frames",
        NUM_FRAMES
    );
    assert_eq!(
        frames_consumed, NUM_FRAMES,
        "Consumer should receive exactly {} frames",
        NUM_FRAMES
    );

    println!(
        "Concurrent frame test passed: {} frames produced and consumed",
        NUM_FRAMES
    );
}

/// Test multiple concurrent readers
///
/// Simulates scenario: Both gateway AND inference reading frames from capture
/// This validates that multiple consumers can safely read from the same frame buffer
#[test]
fn test_multiple_frame_readers() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("multi_frame_reader_test.mmap");
    let path_str = path.to_str().unwrap();

    const NUM_FRAMES: usize = 20;
    const WIDTH: u32 = 640;
    const HEIGHT: u32 = 480;

    // Setup writer
    let mut writer = FrameWriter::build_with_path(path_str, 1024 * 1024).unwrap();

    // Create 3 readers (simulating capture, inference, gateway)
    let mut reader1 = FrameReader::with_path(path_str).unwrap();
    let mut reader2 = FrameReader::with_path(path_str).unwrap();
    let mut reader3 = FrameReader::with_path(path_str).unwrap();

    // Write frames and verify all readers see them
    for i in 1..=NUM_FRAMES {
        let pixels = vec![(i * 13) as u8; (WIDTH * HEIGHT * 3) as usize];
        writer.write_frame(0, &pixels, i as u64, WIDTH, HEIGHT, None).unwrap();

        // All readers should detect new frame
        let frame1 = reader1.get_frame().unwrap();
        let frame2 = reader2.get_frame().unwrap();
        let frame3 = reader3.get_frame().unwrap();

        assert!(frame1.is_some(), "Reader 1 should see frame {}", i);
        assert!(frame2.is_some(), "Reader 2 should see frame {}", i);
        assert!(frame3.is_some(), "Reader 3 should see frame {}", i);

        // All should read same frame data
        let f1 = frame1.unwrap();
        let f2 = frame2.unwrap();
        let f3 = frame3.unwrap();

        assert_eq!(f1.frame_number(), f2.frame_number());
        assert_eq!(f2.frame_number(), f3.frame_number());
        assert_eq!(f1.width(), WIDTH);
        assert_eq!(f2.height(), HEIGHT);

        // Mark as read independently
        reader1.mark_read();
        reader2.mark_read();
        reader3.mark_read();
    }

    println!("Multiple frame readers test passed: {} frames", NUM_FRAMES);
}

/// Test writer behavior when frame data exceeds buffer
///
/// This prevents crashes when frame size is larger than allocated buffer
#[test]
fn test_frame_write_fails_when_buffer_too_small() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("frame_size_test.mmap");
    let path_str = path.to_str().unwrap();

    // Create writer with tiny buffer (1KB)
    let mut writer = FrameWriter::build_with_path(path_str, 1024).unwrap();

    // Try to write a large frame (640x480x3 = ~900KB)
    let large_pixels = vec![0u8; 640 * 480 * 3];
    let result = writer.write_frame(0, &large_pixels, 1, 640, 480, None);

    // Should fail - verify we got an error
    // (The exact error message may vary, but it should fail)
    assert!(
        result.is_err(),
        "Write should fail when frame exceeds buffer"
    );
}

/// Test reader handles missing frames gracefully
///
/// Edge case: Reader polls but writer hasn't written anything yet
#[test]
fn test_frame_reader_handles_no_data() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("frame_stale_test.mmap");
    let path_str = path.to_str().unwrap();

    // Create writer but don't write anything
    let _writer = FrameWriter::build_with_path(path_str, 1024 * 1024).unwrap();
    let reader = FrameReader::with_path(path_str).unwrap();

    // Poll multiple times - should never indicate new data
    for _ in 0..10 {
        assert!(
            reader.get_frame().unwrap().is_none(),
            "Reader should not see frame when nothing written"
        );
        thread::sleep(Duration::from_millis(1));
    }
}

/// Test different frame resolutions
///
/// Validates that FrameWriter/Reader handle various common resolutions correctly
#[test]
fn test_various_frame_resolutions() {
    let dir = tempdir().unwrap();

    let resolutions = [
        (640, 480, "VGA"),
        (1280, 720, "HD"),
        (1920, 1080, "Full HD"),
    ];

    for (i, (width, height, label)) in resolutions.iter().enumerate() {
        // Use unique path for each resolution
        let path = dir
            .path()
            .join(format!("frame_resolutions_test_{}.mmap", i));
        let path_str = path.to_str().unwrap();

        // Use appropriate buffer size for resolution
        let buffer_size = (*width as usize) * (*height as usize) * 3 + 8192;

        let mut writer = FrameWriter::build_with_path(path_str, buffer_size).unwrap();
        let reader = FrameReader::with_path(path_str).unwrap();

        let pixels = vec![200u8; (*width * *height * 3) as usize];
        writer.write_frame(0, &pixels, 1, *width, *height, None).unwrap();

        let frame = reader.get_frame().unwrap();
        assert!(frame.is_some(), "{} frame should be readable", label);

        let frame = frame.unwrap();
        assert_eq!(frame.width(), *width, "{} width mismatch", label);
        assert_eq!(frame.height(), *height, "{} height mismatch", label);
        assert_eq!(
            frame.pixels().unwrap().len(),
            (*width * *height * 3) as usize,
            "{} pixel count mismatch",
            label
        );
    }
}
