use crate::{
    errors::BridgeError, macros::impl_mmap_reader_base, mmap_reader::MmapReader, paths,
    retry::RetryConfig, types::TraceMetadata, utils::safe_flatbuffers_root,
};
use anyhow::Result;
use common::span;

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

    /// Get the current frame along with its trace context for distributed tracing.
    /// Returns None if sequence is 0.
    pub fn get_frame_with_context(
        &self,
    ) -> Result<Option<(schema::Frame<'_>, Option<TraceMetadata>)>> {
        let _s = span!("get_frame_with_context");

        if self.current_sequence() == 0 {
            return Ok(None);
        }

        let frame = safe_flatbuffers_root::<schema::Frame>(self.reader.buffer())?;
        let trace_ctx = extract_trace_context_from_frame(&frame);

        Ok(Some((frame, trace_ctx)))
    }

    /// Get frame with automatic retry using exponential backoff
    ///
    /// Retries when no data is available (sequence == 0), using the provided
    /// retry configuration. Returns the frame on success, or `NoDataAvailable`
    /// error if max attempts are exhausted.
    ///
    /// Deserialization errors are not retried and propagate immediately.
    pub fn get_frame_with_retry(&self, config: &RetryConfig) -> Result<schema::Frame<'_>> {
        for attempt in 0..config.max_attempts {
            match self.get_frame()? {
                Some(frame) => return Ok(frame),
                None => {
                    if attempt < config.max_attempts - 1 {
                        std::thread::sleep(config.delay_for_attempt(attempt));
                    }
                }
            }
        }
        Err(BridgeError::NoDataAvailable.into())
    }

    /// Async version of `get_frame_with_retry` using tokio
    ///
    /// Uses `tokio::time::sleep` instead of blocking thread sleep,
    /// making it suitable for async contexts like the gateway.
    #[cfg(feature = "tokio")]
    pub async fn get_frame_with_retry_async(
        &self,
        config: &RetryConfig,
    ) -> Result<schema::Frame<'_>> {
        for attempt in 0..config.max_attempts {
            match self.get_frame()? {
                Some(frame) => return Ok(frame),
                None => {
                    if attempt < config.max_attempts - 1 {
                        tokio::time::sleep(config.delay_for_attempt(attempt)).await;
                    }
                }
            }
        }
        Err(BridgeError::NoDataAvailable.into())
    }
}

/// Extract trace context from a Frame if present and valid.
fn extract_trace_context_from_frame(frame: &schema::Frame<'_>) -> Option<TraceMetadata> {
    let trace_id = frame.trace_id()?;
    let span_id = frame.span_id()?;

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
        trace_flags: frame.trace_flags(),
    })
}
