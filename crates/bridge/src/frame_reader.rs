use crate::{macros::impl_mmap_reader_base, mmap_reader::MmapReader, paths, utils::safe_flatbuffers_root};
use anyhow::Result;

pub struct FrameReader {
    reader: MmapReader,
}

impl_mmap_reader_base!(FrameReader, paths::FRAME_BUFFER_PATH);

impl FrameReader {
    /// Get the current frame from the buffer with safe deserialization
    /// Returns None if sequence is 0
    pub fn get_frame(&self) -> Result<Option<schema::Frame<'_>>> {
        if self.current_sequence() == 0 {
            return Ok(None);
        }

        let frame = safe_flatbuffers_root::<schema::Frame>(self.reader.buffer())?;
        Ok(Some(frame))
    }
}
