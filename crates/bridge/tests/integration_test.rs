use bridge::{MmapReader, MmapWriter};
use std::thread;
use std::time::Duration;
use tempfile::tempdir;

/// Test basic writer-reader synchronization
///
/// Tests:
/// - Initial state (no data available)
/// - Write visibility to reader
/// - Sequence number progression
/// - mark_read() synchronization
/// - Multiple writes
#[test]
fn test_writer_reader_synchronization() {
    // Setup: Create temporary file for mmap
    let dir = tempdir().unwrap();
    let path = dir.path().join("sync_test.mmap");

    // Initialize writer and reader
    let mut writer = MmapWriter::create_and_init(&path, 4096).unwrap();
    let mut reader = MmapReader::new(&path).unwrap();

    // TEST 1: Initially, reader should have no data
    // Sequence starts at 0, which means "no data written yet"
    assert_eq!(
        reader.last_sequence(),
        0,
        "Reader should start with sequence 0"
    );
    assert_eq!(
        reader.current_sequence(),
        0,
        "Shared memory should have sequence 0"
    );
    assert!(
        reader.has_new_data().is_none(),
        "Reader should not detect data when sequence is 0"
    );

    // TEST 2: Write first frame
    let frame1 = b"frame_001";
    writer.write(frame1).unwrap();

    // Sequence should be incremented
    assert_eq!(
        writer.sequence(),
        1,
        "Writer should have sequence 1 after write"
    );

    // Reader should detect new data
    assert!(
        reader.has_new_data().is_some(),
        "Reader should detect new data (seq 1 > last_seq 0)"
    );
    assert_eq!(
        reader.current_sequence(),
        1,
        "Reader should see sequence 1 in shared memory"
    );

    // Data should be readable
    let buffer = reader.buffer();
    assert_eq!(
        &buffer[..frame1.len()],
        frame1,
        "Reader should read exact data written"
    );

    // TEST 3: Mark as read
    reader.mark_read();
    assert_eq!(
        reader.last_sequence(),
        1,
        "Reader's last_sequence should update to 1"
    );
    assert!(
        reader.has_new_data().is_none(),
        "Reader should not detect new data after mark_read()"
    );

    // TEST 4: Write second frame
    let frame2 = b"frame_002";
    writer.write(frame2).unwrap();

    // Sequence should increment
    assert_eq!(writer.sequence(), 2, "Writer should have sequence 2");

    // Reader should detect new data again
    assert!(
        reader.has_new_data().is_some(),
        "Reader should detect new data (seq 2 > last_seq 1)"
    );

    // Data should be updated
    let buffer = reader.buffer();
    assert_eq!(
        &buffer[..frame2.len()],
        frame2,
        "Reader should read new data"
    );

    // TEST 5: Multiple consecutive writes
    for i in 3..=10 {
        let data = format!("frame_{:03}", i);
        writer.write(data.as_bytes()).unwrap();

        assert_eq!(
            writer.sequence(),
            i as u64,
            "Sequence should increment to {}",
            i
        );
        assert!(
            reader.has_new_data().is_some(),
            "Reader should detect data for frame {}",
            i
        );

        reader.mark_read();
    }

    // Final state check
    assert_eq!(writer.sequence(), 10, "Writer should end at sequence 10");
    assert_eq!(
        reader.last_sequence(),
        10,
        "Reader should end at sequence 10"
    );
}

/// Test concurrent producer-consumer pattern
///
/// This catches race conditions and synchronization bugs.
/// Simulates real-world scenario: gateway writing frames while inference reads them.
///
/// Tests:
/// - Thread safety of atomic operations
/// - Sequence number monotonicity across threads
/// - Consumer doesn't miss frames
#[test]
fn test_concurrent_producer_consumer() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("concurrent_test.mmap");

    // Configuration
    const NUM_FRAMES: u64 = 30;
    const FRAME_SIZE: usize = 256;

    let path_producer = path.clone();
    let path_consumer = path.clone();

    // Producer thread: Initializes file and writes frames
    let producer = thread::spawn(move || {
        let mut writer = MmapWriter::create_and_init(&path_producer, FRAME_SIZE + 8).unwrap();

        // Give consumer time to open file after we initialize it
        thread::sleep(Duration::from_millis(50));

        for i in 1..=NUM_FRAMES {
            // Create frame data with recognizable pattern
            let mut data = vec![0u8; FRAME_SIZE];
            // Embed frame number in first 8 bytes for verification
            data[..8].copy_from_slice(&i.to_le_bytes());

            writer.write(&data).unwrap();

            // Verify sequence increments correctly
            assert_eq!(
                writer.sequence(),
                i,
                "Producer: sequence should match frame number"
            );

            // Small delay to simulate realistic frame rate (~100 FPS)
            thread::sleep(Duration::from_millis(10));
        }

        writer.sequence()
    });

    // Consumer thread: Waits for file then reads all frames
    let consumer = thread::spawn(move || {
        // Wait for producer to create and initialize file
        thread::sleep(Duration::from_millis(20));

        let mut reader = MmapReader::new(&path_consumer).unwrap();
        let mut frames_seen = Vec::new();

        // Keep reading until we've seen the final frame
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

            if reader.has_new_data().is_some() {
                let current_seq = reader.current_sequence();
                let buffer = reader.buffer();

                // Extract frame number from data
                let mut frame_num_bytes = [0u8; 8];
                frame_num_bytes.copy_from_slice(&buffer[..8]);
                let frame_num = u64::from_le_bytes(frame_num_bytes);

                // Verify sequence number matches frame number
                assert_eq!(
                    current_seq, frame_num,
                    "Consumer: sequence should match frame number"
                );

                // Record this frame
                frames_seen.push(frame_num);

                reader.mark_read();
            } else {
                // Poll frequently
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

    // Wait for both threads to complete
    let final_producer_seq = producer.join().expect("Producer thread panicked");
    let frames_consumed = consumer.join().expect("Consumer thread panicked");

    // Verify final state
    assert_eq!(
        final_producer_seq, NUM_FRAMES,
        "Producer should write exactly {} frames",
        NUM_FRAMES
    );
    assert_eq!(
        frames_consumed, NUM_FRAMES,
        "Consumer should receive exactly {} frames",
        NUM_FRAMES
    );

    println!(
        "Concurrent test passed: {} frames produced and consumed successfully",
        NUM_FRAMES
    );
}

/// Test writer behavior when buffer is too small
///
/// This prevents crashes when frame size exceeds allocated buffer
#[test]
fn test_write_fails_when_data_exceeds_buffer() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("size_test.mmap");

    // Create writer with small buffer (100 bytes)
    let mut writer = MmapWriter::create_and_init(&path, 100).unwrap();

    // Try to write data larger than buffer (200 bytes)
    let large_data = vec![0u8; 200];
    let result = writer.write(&large_data);

    // Should return SizeMismatch error
    assert!(
        result.is_err(),
        "Write should fail when data exceeds buffer"
    );

    // Verify error type
    match result {
        Err(bridge::BridgeError::SizeMismatch) => {
            // Expected error
        }
        Err(e) => panic!("Expected SizeMismatch error, got: {:?}", e),
        Ok(_) => panic!("Expected error, but write succeeded"),
    }
}

/// Test reader handles missing sequence updates gracefully
///
/// Edge case: Reader polls but writer hasn't written anything yet
#[test]
fn test_reader_handles_stale_data() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("stale_test.mmap");

    // Create writer but don't write anything
    let _writer = MmapWriter::create_and_init(&path, 1024).unwrap();
    let reader = MmapReader::new(&path).unwrap();

    // Poll multiple times - should never indicate new data
    for _ in 0..10 {
        assert!(
            reader.has_new_data().is_none(),
            "Reader should not see data when nothing written"
        );
        thread::sleep(Duration::from_millis(1));
    }

    // Sequence should remain 0
    assert_eq!(reader.current_sequence(), 0);
    assert_eq!(reader.last_sequence(), 0);
}

/// Test multiple readers can read from same writer
///
/// Simulates scenario: inference and gateway both reading from capture
#[test]
fn test_multiple_concurrent_readers() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("multi_reader_test.mmap");

    const NUM_FRAMES: usize = 50;

    // Setup writer
    let mut writer = MmapWriter::create_and_init(&path, 1024).unwrap();

    // Create 3 readers
    let mut reader1 = MmapReader::new(&path).unwrap();
    let mut reader2 = MmapReader::new(&path).unwrap();
    let mut reader3 = MmapReader::new(&path).unwrap();

    // Write frames and verify all readers see them
    for i in 1..=NUM_FRAMES {
        let data = format!("frame_{}", i);
        writer.write(data.as_bytes()).unwrap();

        // All readers should detect new data
        assert!(
            reader1.has_new_data().is_some(),
            "Reader 1 should see frame {}",
            i
        );
        assert!(
            reader2.has_new_data().is_some(),
            "Reader 2 should see frame {}",
            i
        );
        assert!(
            reader3.has_new_data().is_some(),
            "Reader 3 should see frame {}",
            i
        );

        // All should read same data
        assert_eq!(
            reader1.buffer()[..data.len()],
            reader2.buffer()[..data.len()]
        );
        assert_eq!(
            reader2.buffer()[..data.len()],
            reader3.buffer()[..data.len()]
        );

        // Mark as read independently
        reader1.mark_read();
        reader2.mark_read();
        reader3.mark_read();
    }

    println!("Multiple readers test passed: {} frames", NUM_FRAMES);
}
