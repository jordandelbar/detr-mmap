use crate::{
    macros::impl_mmap_reader_base, mmap_reader::MmapReader, paths, types::TraceMetadata,
    utils::safe_flatbuffers_root,
};
use anyhow::Result;
use common::span;
use schema::Frame;

pub struct FrameReader {
    reader: MmapReader,
}

impl_mmap_reader_base!(FrameReader, paths::FRAME_BUFFER_PATH);

impl FrameReader {
    /// Get the current frame along with its trace context for distributed tracing.
    /// Returns None if sequence is 0.
    pub fn get_frame(&self) -> Result<Option<(Frame<'_>, Option<TraceMetadata>)>> {
        let _s = span!("get_frame_with_context");

        if self.current_sequence() == 0 {
            return Ok(None);
        }

        let frame = safe_flatbuffers_root::<Frame>(self.reader.buffer())?;
        let trace_ctx = extract_trace_context_from_frame(&frame);

        Ok(Some((frame, trace_ctx)))
    }
}

/// Extract trace context from a Frame if present and valid.
fn extract_trace_context_from_frame(frame: &Frame<'_>) -> Option<TraceMetadata> {
    let trace = frame.trace()?;

    Some(TraceMetadata {
        trace_id: std::array::from_fn(|i| trace.trace_id().get(i)),
        span_id: std::array::from_fn(|i| trace.span_id().get(i)),
        trace_flags: trace.trace_flags(),
    })
}
