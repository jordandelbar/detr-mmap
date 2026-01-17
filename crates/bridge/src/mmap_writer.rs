use crate::errors::BridgeError;
use crate::header::Header;
use memmap2::{MmapMut, MmapOptions};
use std::fs::OpenOptions;
use std::path::Path;
use std::sync::atomic::Ordering;

pub(crate) struct MmapWriter {
    mmap: MmapMut,
    sequence: u64,
}

impl MmapWriter {
    /// Create or open an mmap file and reset the sequence to 0.
    ///
    /// Creates the file if it doesn't exist, expands it if undersized.
    /// Resets the sequence number to 0 (readers will wait for new data).
    ///
    /// Use `open_existing()` instead if you want to preserve the sequence.
    pub fn create_and_init(path: impl AsRef<Path>, size: usize) -> Result<Self, BridgeError> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)?;

        // Only resize if the file is smaller than needed
        if file.metadata()?.len() < size as u64 {
            file.set_len(size as u64)?;
        }

        let mut mmap = unsafe { MmapOptions::new().map_mut(&file)? };

        // Initialize sequence number to 0
        let header = unsafe { &mut *(mmap.as_mut_ptr() as *mut Header) };
        header.sequence.store(0, Ordering::Release);

        Ok(Self { mmap, sequence: 0 })
    }

    /// Open an existing mmap file and preserve the sequence number.
    ///
    /// Use this when a writer restarts and you want to continue from where
    /// the previous writer left off. Readers will not miss a beat.
    ///
    /// Returns an error if the file doesn't exist.
    pub fn open_existing(path: impl AsRef<Path>) -> Result<Self, BridgeError> {
        let file = OpenOptions::new().read(true).write(true).open(path)?;

        let mmap = unsafe { MmapOptions::new().map_mut(&file)? };

        // Read current sequence from file (don't reset to 0)
        let header = unsafe { &*(mmap.as_ptr() as *const Header) };
        let sequence = header.sequence.load(Ordering::Acquire);

        Ok(Self { mmap, sequence })
    }

    /// Write data to the buffer and publish with sequence increment.
    ///
    /// Safety: this function ensures correct memory ordering:
    /// 1. Payload is written first via copy_from_slice
    /// 2. Sequence is published with Ordering::Release
    ///
    /// This guarantees readers using Acquire will see the complete payload.
    pub fn write(&mut self, data: &[u8]) -> Result<(), BridgeError> {
        let available_space = self.mmap.len() - Header::SIZE;
        if data.len() > available_space {
            return Err(BridgeError::SizeMismatch);
        }

        // Write payload first
        self.mmap[Header::SIZE..Header::SIZE + data.len()].copy_from_slice(data);

        // Publish with Release ordering (happens-after payload write)
        self.sequence += 1;
        let header = unsafe { &mut *(self.mmap.as_mut_ptr() as *mut Header) };
        header.sequence.store(self.sequence, Ordering::Release);

        Ok(())
    }

    /// Returns mutable buffer for direct writes.
    ///
    /// After writing directly to this buffer, you must manually
    /// publish the sequence number to signal readers. Use with caution.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn buffer_mut(&mut self) -> &mut [u8] {
        &mut self.mmap[Header::SIZE..]
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn flush(&mut self) -> Result<(), BridgeError> {
        self.mmap.flush()?;
        Ok(())
    }

    pub fn sequence(&self) -> u64 {
        self.sequence
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mmap_reader::MmapReader;
    use std::sync::atomic::Ordering;
    use tempfile::NamedTempFile;

    #[test]
    fn test_new_writer_initializes_sequence_to_zero() {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        let writer = MmapWriter::create_and_init(path, 1024).unwrap();

        // Internal sequence should be 0
        assert_eq!(writer.sequence(), 0, "New writer should have sequence = 0");

        // Sequence in mmap should also be 0
        let reader = MmapReader::build(path).unwrap();
        assert_eq!(
            reader.current_sequence(),
            0,
            "Sequence in mmap should be initialized to 0"
        );
    }

    #[test]
    fn test_write_increments_sequence_atomically() {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        let mut writer = MmapWriter::create_and_init(path, 1024).unwrap();

        assert_eq!(writer.sequence(), 0);

        // First write
        writer.write(b"data1").unwrap();
        assert_eq!(
            writer.sequence(),
            1,
            "Sequence should be 1 after first write"
        );

        // Second write
        writer.write(b"data2").unwrap();
        assert_eq!(
            writer.sequence(),
            2,
            "Sequence should be 2 after second write"
        );

        // Verify reader sees the atomic updates
        let reader = MmapReader::build(path).unwrap();
        assert_eq!(
            reader.current_sequence(),
            2,
            "Reader should see sequence = 2"
        );
    }

    #[test]
    fn test_write_copies_data_before_sequence_update() {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        let mut writer = MmapWriter::create_and_init(path, 1024).unwrap();
        let test_data = b"Memory ordering test";

        // Write data
        writer.write(test_data).unwrap();

        // Reader should see both the updated sequence AND the data
        let reader = MmapReader::build(path).unwrap();
        assert_eq!(reader.current_sequence(), 1, "Sequence should be updated");

        let buffer = reader.buffer();
        assert_eq!(
            &buffer[..test_data.len()],
            test_data,
            "Data should be visible after sequence update (memory ordering)"
        );
    }

    #[test]
    fn test_flush_persists_to_disk() {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        let mut writer = MmapWriter::create_and_init(path, 1024).unwrap();

        // Write and flush
        writer.write(b"flushed data").unwrap();
        writer.flush().unwrap();

        // Create a new reader (forces re-reading from disk)
        drop(writer);
        let reader = MmapReader::build(path).unwrap();

        assert_eq!(
            reader.current_sequence(),
            1,
            "Flushed sequence should persist"
        );
        let buffer = reader.buffer();
        assert_eq!(
            &buffer[..12],
            b"flushed data",
            "Flushed data should persist"
        );
    }

    #[test]
    fn test_buffer_mut_allows_direct_writes() {
        use crate::header::Header;

        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        let mut writer = MmapWriter::create_and_init(path, 1024).unwrap();

        // Write directly to buffer
        let buffer = writer.buffer_mut();
        buffer[0] = 42;
        buffer[1] = 43;
        buffer[2] = 44;

        // Manually increment sequence (simulating what a direct write would do)
        writer.sequence += 1;
        let header = unsafe { &mut *(writer.mmap.as_mut_ptr() as *mut Header) };
        header.sequence.store(writer.sequence, Ordering::Release);

        // Verify data is readable
        let reader = MmapReader::build(path).unwrap();
        let read_buffer = reader.buffer();
        assert_eq!(read_buffer[0], 42);
        assert_eq!(read_buffer[1], 43);
        assert_eq!(read_buffer[2], 44);
    }

    #[test]
    fn test_open_existing_preserves_sequence() {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        // Initial writer creates and writes some frames
        {
            let mut writer = MmapWriter::create_and_init(path, 1024).unwrap();
            writer.write(b"frame 1").unwrap();
            writer.write(b"frame 2").unwrap();
            writer.write(b"frame 3").unwrap();
            assert_eq!(writer.sequence(), 3);
        } // Writer drops

        // New writer opens existing file (simulates writer restart)
        let mut writer = MmapWriter::open_existing(path).unwrap();

        // Sequence should be preserved from file
        assert_eq!(
            writer.sequence(),
            3,
            "open_existing should preserve sequence from file"
        );

        // Continue writing from where we left off
        writer.write(b"frame 4").unwrap();
        assert_eq!(writer.sequence(), 4);

        // Reader should see sequence 4
        let reader = MmapReader::build(path).unwrap();
        assert_eq!(reader.current_sequence(), 4);
    }

    #[test]
    fn test_open_existing_safe_with_concurrent_readers() {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        // Initial setup
        let mut writer = MmapWriter::create_and_init(path, 1024).unwrap();
        writer.write(b"initial").unwrap();
        drop(writer);

        // Reader opens file and stays open
        let mut reader = MmapReader::build(path).unwrap();
        assert_eq!(reader.current_sequence(), 1);

        // Writer restarts using open_existing() - should NOT cause SIGBUS
        let mut writer = MmapWriter::open_existing(path).unwrap();

        // Write new data - reader should see it (no crash)
        writer.write(b"new data").unwrap();

        // Reader can still read without SIGBUS
        assert_eq!(reader.current_sequence(), 2);
        reader.mark_read();

        // Multiple reopens should all be safe
        drop(writer);
        let mut writer = MmapWriter::open_existing(path).unwrap();
        writer.write(b"more data").unwrap();

        assert_eq!(reader.current_sequence(), 3);
    }

    #[test]
    fn test_concurrent_producer_consumer() {
        use std::thread;
        use std::time::Duration;

        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path().to_path_buf();

        const NUM_FRAMES: u64 = 30;
        const FRAME_SIZE: usize = 256;

        let path_producer = path.clone();
        let path_consumer = path.clone();

        // Producer thread
        let producer = thread::spawn(move || {
            let mut writer = MmapWriter::create_and_init(&path_producer, FRAME_SIZE + 8).unwrap();
            thread::sleep(Duration::from_millis(50));

            for i in 1..=NUM_FRAMES {
                let mut data = vec![0u8; FRAME_SIZE];
                data[..8].copy_from_slice(&i.to_le_bytes());
                writer.write(&data).unwrap();
                assert_eq!(writer.sequence(), i);
                thread::sleep(Duration::from_millis(10));
            }

            writer.sequence()
        });

        // Consumer thread
        let consumer = thread::spawn(move || {
            thread::sleep(Duration::from_millis(20));
            let mut reader = MmapReader::build(&path_consumer).unwrap();
            let mut frames_seen = Vec::new();

            let start = std::time::Instant::now();
            let timeout = Duration::from_secs(5);

            while frames_seen.len() < NUM_FRAMES as usize {
                if start.elapsed() > timeout {
                    panic!("Consumer timeout: only saw {} frames", frames_seen.len());
                }

                if reader.has_new_data().is_some() {
                    let buffer = reader.buffer();
                    let mut frame_num_bytes = [0u8; 8];
                    frame_num_bytes.copy_from_slice(&buffer[..8]);
                    let frame_num = u64::from_le_bytes(frame_num_bytes);
                    frames_seen.push(frame_num);
                    reader.mark_read();
                } else {
                    thread::sleep(Duration::from_millis(5));
                }
            }

            frames_seen.len() as u64
        });

        let final_producer_seq = producer.join().expect("Producer thread panicked");
        let frames_consumed = consumer.join().expect("Consumer thread panicked");

        assert_eq!(final_producer_seq, NUM_FRAMES);
        assert_eq!(frames_consumed, NUM_FRAMES);
    }
}
