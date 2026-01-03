use std::env;

pub use common::Environment;

#[derive(Debug, Clone)]
pub struct CameraConfig {
    pub environment: Environment,
    pub camera_id: u32,
    pub device_id: u32,
    pub sentry_mode_fps: f64,
    pub mmap_path: String,
    pub mmap_size: usize,
}

impl CameraConfig {
    pub fn from_env() -> anyhow::Result<Self> {
        let environment = Environment::from_env();

        let camera_id = env::var("CAMERA_ID")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);

        let device_id = env::var("DEVICE_ID")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);

        let sentry_mode_fps = env::var("SENTRY_MODE_FPS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1.0);

        let mmap_path = env::var("FRAME_MMAP_PATH")
            .unwrap_or_else(|_| "/dev/shm/bridge_frame_buffer".to_string());

        let mmap_size = env::var("DETECTION_MMAP_SIZE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(32 * 1024 * 1024);

        Ok(Self {
            environment,
            camera_id,
            device_id,
            sentry_mode_fps,
            mmap_path,
            mmap_size,
        })
    }
}
