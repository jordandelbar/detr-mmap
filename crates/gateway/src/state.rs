use bridge::Detection;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::broadcast;

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
