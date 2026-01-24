use crate::{
    macros::impl_mmap_reader_base, mmap_reader::MmapReader, paths, utils::safe_flatbuffers_root,
};
use anyhow::Result;
use common::span;
use schema::Frame;

pub struct FrameReader {
    reader: MmapReader,
}

impl_mmap_reader_base!(FrameReader, paths::FRAME_BUFFER_PATH);

impl FrameReader {
    /// Get the current frame from shared memory.
    /// Returns None if sequence is 0.
    ///
    /// The trace context can be accessed via `frame.trace()` when needed for distributed tracing.
    pub fn get_frame(&self) -> Result<Option<Frame<'_>>> {
        let _s = span!("get_frame");

        if self.current_sequence() == 0 {
            return Ok(None);
        }

        let frame = safe_flatbuffers_root::<Frame>(self.reader.buffer())?;
        Ok(Some(frame))
    }
}
