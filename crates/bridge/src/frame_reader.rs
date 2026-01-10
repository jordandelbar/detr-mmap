use crate::MmapReader;
use anyhow::Result;

pub struct FrameReader {
    reader: MmapReader,
}

/// Safely deserialize flatbuffers with bounds checking
fn safe_flatbuffers_root<'a, T>(buffer: &'a [u8]) -> anyhow::Result<T::Inner>
where
    T: flatbuffers::Follow<'a> + flatbuffers::Verifiable + 'a,
{
    // Check minimum buffer size
    if buffer.len() < 8 {
        return Err(anyhow::anyhow!(
            "Buffer too small for flatbuffers: {} bytes",
            buffer.len()
        ));
    }

    // Attempt to get the root with proper error handling
    match flatbuffers::root::<T>(buffer) {
        Ok(root) => Ok(root),
        Err(e) => Err(anyhow::anyhow!(
            "Flatbuffers deserialization failed: {:?}, buffer size: {}",
            e,
            buffer.len()
        )),
    }
}

impl FrameReader {
    pub fn build(mmap_path: &str) -> Result<Self> {
        let reader = MmapReader::build(mmap_path)?;

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
