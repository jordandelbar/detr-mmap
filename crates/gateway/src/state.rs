use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::broadcast;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Detection {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub confidence: f32,
    pub class_id: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameMessage {
    pub frame_number: u64,
    pub timestamp_ns: u64,
    pub width: u32,
    pub height: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detections: Option<Vec<Detection>>,
    pub status: String,
}

#[derive(Clone)]
pub struct FramePacket {
    pub metadata: FrameMessage,
    pub jpeg_data: Vec<u8>,
}

#[derive(Clone)]
pub struct AppState {
    pub tx: Arc<broadcast::Sender<FramePacket>>,
}
