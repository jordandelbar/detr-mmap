use crate::errors::BridgeError;
use crate::header::Header;
use memmap2::{MmapMut, MmapOptions};
use std::fs::OpenOptions;
use std::path::Path;
use std::sync::atomic::Ordering;

pub struct FrameWriter {
    mmap: MmapMut,
    sequence: u64,
}

impl FrameWriter {
    /// Create a new mmap file and initialize it for IPC.
    ///
    /// Critical safety requirement:
    /// This method uses `truncate(true)` which will cause sigbus in concurrent readers
    ///
    /// On Linux, if readers have this file mmap'ed when truncate happens:
    /// - Readers will crash with SIGBUS
    /// - Undefined behavior at runtime
    /// - Data corruption possible
    ///
    /// Only use this when:
    /// - System is starting up (no readers exist yet)
    /// - In tests where you control the lifecycle
    /// - You can guarantee no concurrent readers
    ///
    /// Production safe pattern:
    /// ```no_run
    /// # use bridge::FrameWriter;
    /// // At system init (no readers yet), it's safe
    /// let writer = FrameWriter::create_and_init("/dev/shm/frames", 4096).unwrap();
    ///
    /// // Later, writer restarts but readers may be active, use open_existing()
    /// let writer = FrameWriter::open_existing("/dev/shm/frames").unwrap();
    /// ```
    pub fn create_and_init(path: impl AsRef<Path>, size: usize) -> Result<Self, BridgeError> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true) // SIGBUS danger with concurrent readers
            .open(path)?;

        file.set_len(size as u64)?;

        let mut mmap = unsafe { MmapOptions::new().map_mut(&file)? };

        // Initialize sequence number to 0
        let header = unsafe { &mut *(mmap.as_mut_ptr() as *mut Header) };
        header.sequence.store(0, Ordering::Release);

        Ok(Self { mmap, sequence: 0 })
    }

    /// Open an existing mmap file for writing (safe with concurrent readers).
    ///
    /// Safe with concurrent readers:
    /// - Does not truncate the file
    /// - Will not cause SIGBUS
    /// - Readers can remain active
    ///
    /// Use this when:
    /// - Writer process restarts in production
    /// - Hot-swapping the writer
    /// - Any time readers might have the file open
    ///
    /// Requirements:
    /// - File must already exist (created by `create_and_init()`)
    /// - File must be properly initialized with header
    ///
    /// Example:
    /// ```no_run
    /// # use bridge::FrameWriter;
    /// // Safe even if readers are actively reading:
    /// let mut writer = FrameWriter::open_existing("/dev/shm/frames").unwrap();
    /// writer.write(b"new frame").unwrap();
    /// ```
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
    pub fn buffer_mut(&mut self) -> &mut [u8] {
        &mut self.mmap[Header::SIZE..]
    }

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

        let writer = FrameWriter::create_and_init(path, 1024).unwrap();

        // Internal sequence should be 0
        assert_eq!(writer.sequence(), 0, "New writer should have sequence = 0");

        // Sequence in mmap should also be 0
        let reader = MmapReader::new(path).unwrap();
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

        let mut writer = FrameWriter::create_and_init(path, 1024).unwrap();

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
        let reader = MmapReader::new(path).unwrap();
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

        let mut writer = FrameWriter::create_and_init(path, 1024).unwrap();
        let test_data = b"Memory ordering test";

        // Write data
        writer.write(test_data).unwrap();

        // Reader should see both the updated sequence AND the data
        let reader = MmapReader::new(path).unwrap();
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

        let mut writer = FrameWriter::create_and_init(path, 1024).unwrap();

        // Write and flush
        writer.write(b"flushed data").unwrap();
        writer.flush().unwrap();

        // Create a new reader (forces re-reading from disk)
        drop(writer);
        let reader = MmapReader::new(path).unwrap();

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

        let mut writer = FrameWriter::create_and_init(path, 1024).unwrap();

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
        let reader = MmapReader::new(path).unwrap();
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
            let mut writer = FrameWriter::create_and_init(path, 1024).unwrap();
            writer.write(b"frame 1").unwrap();
            writer.write(b"frame 2").unwrap();
            writer.write(b"frame 3").unwrap();
            assert_eq!(writer.sequence(), 3);
        } // Writer drops

        // New writer opens existing file (simulates writer restart)
        let mut writer = FrameWriter::open_existing(path).unwrap();

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
        let reader = MmapReader::new(path).unwrap();
        assert_eq!(reader.current_sequence(), 4);
    }

    #[test]
    fn test_open_existing_safe_with_concurrent_readers() {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        // Initial setup
        let mut writer = FrameWriter::create_and_init(path, 1024).unwrap();
        writer.write(b"initial").unwrap();
        drop(writer);

        // Reader opens file and stays open
        let mut reader = MmapReader::new(path).unwrap();
        assert_eq!(reader.current_sequence(), 1);

        // Writer restarts using open_existing() - should NOT cause SIGBUS
        let mut writer = FrameWriter::open_existing(path).unwrap();

        // Write new data - reader should see it (no crash)
        writer.write(b"new data").unwrap();

        // Reader can still read without SIGBUS
        assert_eq!(reader.current_sequence(), 2);
        reader.mark_read();

        // Multiple reopens should all be safe
        drop(writer);
        let mut writer = FrameWriter::open_existing(path).unwrap();
        writer.write(b"more data").unwrap();

        assert_eq!(reader.current_sequence(), 3);
    }
}
