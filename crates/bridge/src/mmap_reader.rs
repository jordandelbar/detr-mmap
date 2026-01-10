use crate::errors::BridgeError;
use crate::header::Header;
use memmap2::{Mmap, MmapOptions};
use std::fs::File;
use std::path::Path;
use std::sync::atomic::Ordering;

pub struct MmapReader {
    _file: File,
    mmap: Mmap,
    last_sequence: u64,
}

impl MmapReader {
    pub fn build(path: impl AsRef<Path>) -> Result<Self, BridgeError> {
        let file = File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };

        Ok(Self {
            _file: file,
            mmap,
            last_sequence: 0,
        })
    }

    /// Returns the current sequence number from mmap
    ///
    /// SAFETY: Uses Ordering::Acquire to ensure all payload writes
    /// are visible after observing a new sequence number.
    pub fn current_sequence(&self) -> u64 {
        let header = unsafe { &*(self.mmap.as_ptr() as *const Header) };
        header.sequence.load(Ordering::Acquire)
    }

    /// Checks if new data is available and returns the new sequence if so.
    ///
    /// Returns Some(seq) if there is new data, None otherwise.
    /// This avoids double-loading the sequence number.
    pub fn has_new_data(&self) -> Option<u64> {
        let seq = self.current_sequence();
        if seq > self.last_sequence {
            Some(seq)
        } else {
            None
        }
    }

    /// Returns data buffer (skips the header)
    pub fn buffer(&self) -> &[u8] {
        &self.mmap[Header::SIZE..]
    }

    /// Returns the latest fully-published frame.
    /// The returned buffer may be newer than the returned sequence.
    /// Frames may be skipped.
    pub fn read_frame(&self) -> Option<(u64, &[u8])> {
        let seq1 = self.current_sequence();
        if seq1 <= self.last_sequence {
            return None;
        }
        let buf = self.buffer();
        let seq2 = self.current_sequence();
        if seq1 != seq2 {
            // Sequence changed during read - torn read detected
            return None;
        }
        Some((seq1, buf))
    }

    /// Mark current sequence as read
    pub fn mark_read(&mut self) {
        self.last_sequence = self.current_sequence();
    }

    /// Mark a specific sequence as read
    pub fn mark_read_seq(&mut self, seq: u64) {
        self.last_sequence = seq;
    }

    /// Get last read sequence number
    pub fn last_sequence(&self) -> u64 {
        self.last_sequence
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mmap_writer::MmapWriter;
    use std::sync::Arc;
    use std::thread;
    use tempfile::NamedTempFile;

    #[test]
    fn test_new_reader_starts_with_zero_sequence() {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        // Create a writer to initialize the file
        let _writer = MmapWriter::create_and_init(path, 1024).unwrap();

        // Create reader and verify it starts with sequence 0
        let reader = MmapReader::build(path).unwrap();
        assert_eq!(
            reader.last_sequence(),
            0,
            "New reader should start with last_sequence = 0"
        );
    }

    #[test]
    fn test_has_new_data_returns_none_when_sequence_zero() {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        // Create a writer but don't write any data
        let _writer = MmapWriter::create_and_init(path, 1024).unwrap();

        // Reader should report no new data when sequence is 0
        let reader = MmapReader::build(path).unwrap();
        assert!(
            reader.has_new_data().is_none(),
            "has_new_data should return None when sequence is 0"
        );
    }

    #[test]
    fn test_has_new_data_returns_sequence_after_increment() {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        let mut writer = MmapWriter::create_and_init(path, 1024).unwrap();
        let reader = MmapReader::build(path).unwrap();

        // Initially no new data
        assert!(reader.has_new_data().is_none());

        // Write data (increments sequence from 0 to 1)
        writer.write(&[1, 2, 3, 4]).unwrap();

        // Now there should be new data with sequence 1
        assert_eq!(
            reader.has_new_data(),
            Some(1),
            "has_new_data should return Some(1) after write"
        );
    }

    #[test]
    fn test_mark_read_updates_last_sequence() {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        let mut writer = MmapWriter::create_and_init(path, 1024).unwrap();
        let mut reader = MmapReader::build(path).unwrap();

        // Write data (sequence becomes 1)
        writer.write(&[1, 2, 3]).unwrap();

        assert_eq!(
            reader.last_sequence(),
            0,
            "Initial last_sequence should be 0"
        );

        // Mark as read
        reader.mark_read();

        assert_eq!(
            reader.last_sequence(),
            1,
            "last_sequence should be 1 after mark_read"
        );
        assert!(
            reader.has_new_data().is_none(),
            "has_new_data should return None after marking current sequence as read"
        );
    }

    #[test]
    fn test_buffer_skips_header_bytes() {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        let mut writer = MmapWriter::create_and_init(path, 1024).unwrap();

        // Write known data
        let test_data = b"Hello, World!";
        writer.write(test_data).unwrap();

        let reader = MmapReader::build(path).unwrap();
        let buffer = reader.buffer();

        // Buffer should skip the 8-byte header and start with our data
        assert_eq!(
            &buffer[..test_data.len()],
            test_data,
            "Buffer should skip header and return data starting at offset 8"
        );
    }

    #[test]
    fn test_read_frame_returns_none_for_no_new_data() {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        let _writer = MmapWriter::create_and_init(path, 1024).unwrap();
        let reader = MmapReader::build(path).unwrap();

        // No data written yet
        assert!(
            reader.read_frame().is_none(),
            "read_frame should return None when no data"
        );
    }

    #[test]
    fn test_read_frame_returns_sequence_and_buffer() {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        let mut writer = MmapWriter::create_and_init(path, 1024).unwrap();
        let mut reader = MmapReader::build(path).unwrap();

        // Write data
        writer.write(b"test data").unwrap();

        // read_frame should return the data
        let result = reader.read_frame();
        assert!(
            result.is_some(),
            "read_frame should return Some when data available"
        );

        let (seq, buf) = result.unwrap();
        assert_eq!(seq, 1, "Sequence should be 1");
        assert_eq!(
            &buf[..9],
            b"test data",
            "Buffer should contain the written data"
        );

        // Mark as read
        reader.mark_read_seq(seq);

        // Should return None after marking as read
        assert!(
            reader.read_frame().is_none(),
            "read_frame should return None after data is marked read"
        );
    }

    #[test]
    fn test_concurrent_reads_during_writes_are_consistent() {
        use std::sync::Barrier;

        let temp_file = NamedTempFile::new().unwrap();
        let path = Arc::new(temp_file.path().to_path_buf());

        // Initialize file with proper size
        {
            let writer = MmapWriter::create_and_init(path.as_ref(), 1024).unwrap();
            drop(writer);
        }

        // Barrier to ensure all threads start at roughly the same time
        let barrier = Arc::new(Barrier::new(5)); // 1 writer + 4 readers

        // Writer continuously updates the buffer
        let writer_path = Arc::clone(&path);
        let writer_barrier = Arc::clone(&barrier);
        let writer_handle = thread::spawn(move || {
            let mut writer = MmapWriter::open_existing(writer_path.as_ref()).unwrap();
            writer_barrier.wait(); // Wait for all threads to be ready

            for i in 1..=100 {
                let data = format!("frame-{:03}", i);
                writer.write(data.as_bytes()).unwrap();
                thread::sleep(std::time::Duration::from_micros(100));
            }
        });

        // Multiple readers observe the writes concurrently
        let readers: Vec<_> = (0..4)
            .map(|_| {
                let path = Arc::clone(&path);
                let barrier = Arc::clone(&barrier);
                thread::spawn(move || {
                    let mut reader = MmapReader::build(path.as_ref()).unwrap();
                    barrier.wait(); // Wait for all threads to be ready

                    let mut last_seq = 0;
                    let mut reads = 0;

                    // Poll for updates
                    while last_seq < 100 && reads < 10000 {
                        // Use read_frame() which handles all invariants internally
                        if let Some((seq, buf)) = reader.read_frame() {
                            // INVARIANT 1: Sequence must be monotonically increasing
                            assert!(
                                seq > last_seq,
                                "Sequence must increase: {} -> {}",
                                last_seq,
                                seq
                            );

                            // INVARIANT 2: Buffer must contain valid data
                            let data_str = std::str::from_utf8(&buf[..9]).unwrap();
                            assert!(
                                data_str.starts_with("frame-"),
                                "Buffer must contain valid frame data: {}",
                                data_str
                            );

                            // INVARIANT 3: Payload must be at least as fresh as sequence
                            // (buffer may be newer due to concurrent writes - this is OK for video)
                            let frame_num: u32 = data_str[6..9].parse().unwrap();
                            assert!(
                                frame_num as u64 >= seq,
                                "Buffer older than sequence: seq={}, frame={} (broken publish ordering!)",
                                seq,
                                frame_num
                            );

                            reader.mark_read_seq(seq);
                            last_seq = seq;
                        }
                        reads += 1;
                        thread::sleep(std::time::Duration::from_micros(50));
                    }

                    // Should have observed at least some updates
                    assert!(last_seq > 0, "Reader never saw any updates");
                })
            })
            .collect();

        writer_handle.join().unwrap();
        for handle in readers {
            handle.join().unwrap();
        }
    }
}
