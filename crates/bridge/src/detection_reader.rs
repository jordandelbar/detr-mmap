use crate::{mmap_reader::MmapReader, paths, types::Detection};
use anyhow::Result;

pub struct DetectionReader {
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

impl DetectionReader {
    /// Create a new DetectionReader using the default detection buffer path
    pub fn build() -> Result<Self> {
        Self::with_path(paths::DETECTION_BUFFER_PATH)
    }

    /// Create a new DetectionReader with a custom path (useful for tests)
    pub fn with_path(detection_mmap_path: &str) -> Result<Self> {
        let reader = MmapReader::build(detection_mmap_path)?;
        Ok(Self { reader })
    }

    /// Get the current sequence number of the detection buffer
    pub fn current_sequence(&self) -> u64 {
        self.reader.current_sequence()
    }

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

    /// Mark the current buffer as read
    pub fn mark_read(&mut self) {
        self.reader.mark_read();
    }
}
