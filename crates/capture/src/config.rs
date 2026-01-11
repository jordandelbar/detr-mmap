use std::env;

pub use common::Environment;

#[derive(Debug, Clone)]
pub struct CameraConfig {
    pub environment: Environment,
    pub camera_id: u32,
    pub device_id: u32,
    pub sentry_mode_fps: f64,
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
            .unwrap_or(3.0);

        Ok(Self {
            environment,
            camera_id,
            device_id,
            sentry_mode_fps,
        })
    }
}
