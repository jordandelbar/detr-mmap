use std::env;

pub use common::Environment;

#[derive(Debug, Clone)]
pub struct LogicConfig {
    pub environment: Environment,
    pub frame_mmap_path: String,
    pub detection_mmap_path: String,
    pub poll_interval_ms: u64,
    pub ws_addr: String,
    pub channel_capacity: usize,
}

impl LogicConfig {
    /// Load configuration from environment variables with sensible defaults
    pub fn from_env() -> Self {
        let environment = Environment::from_env();

        let frame_mmap_path = env::var("LOGIC_FRAME_MMAP_PATH")
            .unwrap_or_else(|_| "/dev/shm/bridge_frame_buffer".to_string());

        let detection_mmap_path = env::var("LOGIC_DETECTION_MMAP_PATH")
            .unwrap_or_else(|_| "/dev/shm/bridge_detection_buffer".to_string());

        let poll_interval_ms = env::var("LOGIC_POLL_INTERVAL_MS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(16); // ~60fps

        let ws_addr = env::var("LOGIC_WS_ADDR")
            .unwrap_or_else(|_| "0.0.0.0:8080".to_string());

        let channel_capacity = env::var("LOGIC_CHANNEL_CAPACITY")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(10);

        Self {
            environment,
            frame_mmap_path,
            detection_mmap_path,
            poll_interval_ms,
            ws_addr,
            channel_capacity,
        }
    }

    /// Create default configuration for testing
    #[cfg(test)]
    pub fn default() -> Self {
        Self {
            environment: Environment::Development,
            frame_mmap_path: "/dev/shm/bridge_frame_buffer".to_string(),
            detection_mmap_path: "/dev/shm/bridge_detection_buffer".to_string(),
            poll_interval_ms: 16,
            ws_addr: "0.0.0.0:8080".to_string(),
            channel_capacity: 10,
        }
    }
}
