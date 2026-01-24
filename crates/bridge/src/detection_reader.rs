use crate::{
    macros::impl_mmap_reader_base, mmap_reader::MmapReader, paths, types::TraceMetadata,
    utils::safe_flatbuffers_root,
};
use anyhow::Result;
use common::span;
use schema::DetectionResult;

pub struct DetectionReader {
    reader: MmapReader,
}

impl_mmap_reader_base!(DetectionReader, paths::DETECTION_BUFFER_PATH);

impl DetectionReader {
    /// Get detections from the buffer.
    /// Returns None if sequence is 0.
    pub fn get_detections(&self) -> Result<Option<(DetectionResult<'_>, Option<TraceMetadata>)>> {
        let _s = span!("get_detections");

        if self.current_sequence() == 0 {
            return Ok(None);
        }

        let detection_result = safe_flatbuffers_root::<DetectionResult>(self.reader.buffer())?;
        let trace_ctx = extract_trace_context(&detection_result);

        Ok(Some((detection_result, trace_ctx)))
    }

    /// Check if a person (class_id == 0) is detected in the current buffer
    pub fn check_person_detected(&self) -> Result<bool> {
        if self.current_sequence() == 0 {
            return Ok(false);
        }

        let detection = safe_flatbuffers_root::<DetectionResult>(self.reader.buffer())?;

        if let Some(detections) = detection.detections() {
            for det in detections {
                if det.class_id() == 0 {
                    return Ok(true);
                }
            }
        }

        Ok(false)
    }
}

/// Extract trace context from a DetectionResult if present and valid.
fn extract_trace_context(detection: &DetectionResult<'_>) -> Option<TraceMetadata> {
    let trace = detection.trace()?;

    Some(TraceMetadata {
        trace_id: std::array::from_fn(|i| trace.trace_id().get(i)),
        span_id: std::array::from_fn(|i| trace.span_id().get(i)),
        trace_flags: trace.trace_flags(),
    })
}
