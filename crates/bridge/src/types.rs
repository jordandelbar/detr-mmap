use serde::{Deserialize, Serialize};

/// Detection result with bounding box coordinates, confidence, and class.
/// Used for JSON serialization at API boundaries (gateway).
/// Maps to the FlatBuffers `Detection` table in the schema.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Detection {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub confidence: f32,
    pub class_id: u16,
}

impl From<&schema::Detection<'_>> for Detection {
    fn from(det: &schema::Detection) -> Self {
        let bbox = det.box_().unwrap();
        Self {
            x1: bbox.x1(),
            y1: bbox.y1(),
            x2: bbox.x2(),
            y2: bbox.y2(),
            confidence: det.confidence(),
            class_id: det.class_id(),
        }
    }
}
