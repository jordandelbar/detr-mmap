use crate::{
    errors::BridgeError, macros::impl_mmap_reader_base, mmap_reader::MmapReader, paths,
    retry::RetryConfig,
    types::{Detection, TraceContextBytes},
    utils::safe_flatbuffers_root,
};
use anyhow::Result;

pub struct DetectionReader {
    reader: MmapReader,
}

impl_mmap_reader_base!(DetectionReader, paths::DETECTION_BUFFER_PATH);

impl DetectionReader {
    /// Get all detections from the buffer with safe deserialization
    /// Returns None if sequence is 0 or on deserialization error
    pub fn get_detections(&self) -> Result<Option<Vec<Detection>>> {
        if self.current_sequence() == 0 {
            return Ok(None);
        }

        let detection_result =
            safe_flatbuffers_root::<schema::DetectionResult>(self.reader.buffer())?;

        let detections = detection_result
            .detections()
            .map(|d| d.iter().map(|det| Detection::from(&det)).collect());

        Ok(detections)
    }

    /// Get all detections along with trace context for distributed tracing.
    /// Returns None if sequence is 0.
    pub fn get_detections_with_context(
        &self,
    ) -> Result<Option<(Vec<Detection>, Option<TraceContextBytes>)>> {
        if self.current_sequence() == 0 {
            return Ok(None);
        }

        let detection_result =
            safe_flatbuffers_root::<schema::DetectionResult>(self.reader.buffer())?;

        let detections = detection_result
            .detections()
            .map(|d| d.iter().map(|det| Detection::from(&det)).collect())
            .unwrap_or_default();

        let trace_ctx = extract_trace_context_from_detection(&detection_result);

        Ok(Some((detections, trace_ctx)))
    }

    /// Check if a person (class_id == 0) is detected in the current buffer
    pub fn check_person_detected(&self) -> Result<bool> {
        if self.current_sequence() == 0 {
            return Ok(false);
        }

        let detection = safe_flatbuffers_root::<schema::DetectionResult>(self.reader.buffer())?;

        if let Some(detections) = detection.detections() {
            for det in detections {
                if det.class_id() == 0 {
                    return Ok(true);
                }
            }
        }

        Ok(false)
    }

    /// Get detections with automatic retry using exponential backoff
    ///
    /// Retries when no data is available (sequence == 0), using the provided
    /// retry configuration. Returns detections on success, or `NoDataAvailable`
    /// error if max attempts are exhausted.
    ///
    /// Deserialization errors are not retried and propagate immediately.
    pub fn get_detections_with_retry(
        &self,
        config: &RetryConfig,
    ) -> Result<Option<Vec<Detection>>> {
        for attempt in 0..config.max_attempts {
            match self.get_detections()? {
                Some(detections) => return Ok(Some(detections)),
                None => {
                    if attempt < config.max_attempts - 1 {
                        std::thread::sleep(config.delay_for_attempt(attempt));
                    }
                }
            }
        }
        Err(BridgeError::NoDataAvailable.into())
    }

    /// Async version of `get_detections_with_retry` using tokio
    ///
    /// Uses `tokio::time::sleep` instead of blocking thread sleep,
    /// making it suitable for async contexts like the gateway.
    #[cfg(feature = "tokio")]
    pub async fn get_detections_with_retry_async(
        &self,
        config: &RetryConfig,
    ) -> Result<Option<Vec<Detection>>> {
        for attempt in 0..config.max_attempts {
            match self.get_detections()? {
                Some(detections) => return Ok(Some(detections)),
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

/// Extract trace context from a DetectionResult if present and valid.
fn extract_trace_context_from_detection(
    detection: &schema::DetectionResult<'_>,
) -> Option<TraceContextBytes> {
    let trace_id = detection.trace_id()?;
    let span_id = detection.span_id()?;

    if trace_id.len() != 16 || span_id.len() != 8 {
        return None;
    }

    let mut tid = [0u8; 16];
    let mut sid = [0u8; 8];
    tid.copy_from_slice(trace_id.bytes());
    sid.copy_from_slice(span_id.bytes());

    Some(TraceContextBytes {
        trace_id: tid,
        span_id: sid,
        trace_flags: detection.trace_flags(),
    })
}
