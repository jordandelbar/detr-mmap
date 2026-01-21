use common::{Environment, get_env, get_env_opt};

#[derive(Debug, Clone)]
pub struct CameraConfig {
    pub environment: Environment,
    pub camera_id: u32,
    pub device_id: u32,
    pub sentry_mode_fps: f64,
    pub otel_endpoint: Option<String>,
}

impl CameraConfig {
    pub fn from_env() -> anyhow::Result<Self> {
        Ok(Self {
            environment: Environment::from_env(),
            camera_id: get_env("CAMERA_ID", 0),
            device_id: get_env("DEVICE_ID", 0),
            sentry_mode_fps: get_env("SENTRY_MODE_FPS", 3.0),
            otel_endpoint: get_env_opt("OTEL_EXPORTER_OTLP_ENDPOINT"),
        })
    }
}
