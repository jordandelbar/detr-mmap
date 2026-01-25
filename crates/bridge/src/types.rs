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

impl TryFrom<&schema::Detection<'_>> for Detection {
    type Error = &'static str;

    fn try_from(det: &schema::Detection) -> Result<Self, Self::Error> {
        let bbox = det.box_().ok_or("Detection missing bounding box")?;
        Ok(Self {
            x1: bbox.x1(),
            y1: bbox.y1(),
            x2: bbox.x2(),
            y2: bbox.y2(),
            confidence: det.confidence(),
            class_id: det.class_id(),
        })
    }
}
