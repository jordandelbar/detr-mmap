use crate::{macros::impl_mmap_reader_base, mmap_reader::MmapReader, paths, types::Detection, utils::safe_flatbuffers_root};
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
}
