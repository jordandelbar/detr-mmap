use std::env;

#[derive(Debug, Clone)]
pub struct InferenceConfig {
    pub model_path: String,
    pub frame_mmap_path: String,
    pub detection_mmap_path: String,
    pub detection_mmap_size: usize,
    pub input_size: (u32, u32),
    pub poll_interval_ms: u64,
}

impl InferenceConfig {
    /// Load configuration from environment variables with sensible defaults
    pub fn from_env() -> anyhow::Result<Self> {
        let model_path = env::var("MODEL_PATH")
            .unwrap_or_else(|_| "/models/model.onnx".to_string());

        let frame_mmap_path = env::var("FRAME_MMAP_PATH")
            .unwrap_or_else(|_| "/dev/shm/bridge_frame_buffer".to_string());

        let detection_mmap_path = env::var("DETECTION_MMAP_PATH")
            .unwrap_or_else(|_| "/dev/shm/bridge_detection_buffer".to_string());

        let detection_mmap_size = env::var("DETECTION_MMAP_SIZE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1024 * 1024); // 1MB default

        let input_width = env::var("INPUT_WIDTH")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(640);

        let input_height = env::var("INPUT_HEIGHT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(640);

        let poll_interval_ms = env::var("POLL_INTERVAL_MS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(100);

        Ok(Self {
            model_path,
            frame_mmap_path,
            detection_mmap_path,
            detection_mmap_size,
            input_size: (input_width, input_height),
            poll_interval_ms,
        })
    }

    /// Create default configuration for testing
    #[cfg(test)]
    pub fn default() -> Self {
        Self {
            model_path: "/models/model.onnx".to_string(),
            frame_mmap_path: "/dev/shm/bridge_frame_buffer".to_string(),
            detection_mmap_path: "/dev/shm/bridge_detection_buffer".to_string(),
            detection_mmap_size: 1024 * 1024,
            input_size: (640, 640),
            poll_interval_ms: 100,
        }
    }
}
