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

        let frame = safe_flatbuffers_root::<schema::Frame>(self.reader.buffer())?;
        let trace_ctx = extract_trace_context_from_frame(&frame);

        Ok(Some((frame, trace_ctx)))
    }
}

/// Extract trace context from a Frame if present and valid.
fn extract_trace_context_from_frame(frame: &schema::Frame<'_>) -> Option<TraceMetadata> {
    let trace = frame.trace()?;
    let trace_id = trace.trace_id();
    let span_id = trace.span_id();

    if trace_id.len() != 16 || span_id.len() != 8 {
        return None;
    }

    let mut tid = [0u8; 16];
    let mut sid = [0u8; 8];
    tid.copy_from_slice(trace_id.bytes());
    sid.copy_from_slice(span_id.bytes());

    Some(TraceMetadata {
        trace_id: tid,
        span_id: sid,
        trace_flags: trace.trace_flags(),
    })
}
