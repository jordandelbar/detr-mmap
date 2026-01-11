use crate::{mmap_reader::MmapReader, paths, utils::safe_flatbuffers_root};
use anyhow::Result;

pub struct FrameReader {
    reader: MmapReader,
}

impl FrameReader {
    pub fn build() -> Result<Self> {
        Self::with_path(paths::FRAME_BUFFER_PATH)
    }

    /// Create a new FrameReader with a custom path (useful for tests and benchmarks)
    pub fn with_path(frame_mmap_path: &str) -> Result<Self> {
        let reader = MmapReader::build(frame_mmap_path)?;
        Ok(Self { reader })
    }

    /// Get the current sequence number of the frame buffer
    pub fn current_sequence(&self) -> u64 {
        self.reader.current_sequence()
    }

    /// Get the current frame from the buffer with safe deserialization
    /// Returns None if sequence is 0
    pub fn get_frame(&self) -> Result<Option<schema::Frame<'_>>> {
        if self.current_sequence() == 0 {
            return Ok(None);
        }

        let frame = safe_flatbuffers_root::<schema::Frame>(self.reader.buffer())?;
        Ok(Some(frame))
    }

    /// Mark the current buffer as read
    pub fn mark_read(&mut self) {
        self.reader.mark_read();
    }
}
