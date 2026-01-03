use anyhow::Result;
use bridge::MmapReader;

pub struct DetectionReader {
    reader: MmapReader,
}

impl DetectionReader {
    pub fn new(detection_mmap_path: &str) -> Result<Self> {
        let reader = MmapReader::new(detection_mmap_path)?;
        Ok(Self { reader })
    }

    pub fn check_person_detected(&self) -> Result<bool> {
        let sequence = self.reader.current_sequence();

        if sequence == 0 {
            return Ok(false);
        }

        let detection = flatbuffers::root::<schema::DetectionResult>(self.reader.buffer())?;

        if let Some(detections) = detection.detections() {
            for det in detections {
                if det.class_id() == 0 {
                    return Ok(true);
                }
            }
        }

        Ok(false)
    }

    pub fn mark_read(&mut self) {
        self.reader.mark_read();
    }
}
