use crate::{macros::impl_mmap_reader_base, mmap_reader::MmapReader, paths, utils::safe_flatbuffers_root};
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
    ///
    /// The trace context can be accessed via `result.trace()` when needed for distributed tracing.
    pub fn get_detections(&self) -> Result<Option<DetectionResult<'_>>> {
        let _s = span!("get_detections");

        if self.current_sequence() == 0 {
            return Ok(None);
        }

        let detection_result = safe_flatbuffers_root::<DetectionResult>(self.reader.buffer())?;
        Ok(Some(detection_result))
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
