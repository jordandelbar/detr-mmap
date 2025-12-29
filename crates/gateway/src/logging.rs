use crate::config::CameraConfig;

pub fn setup_logging(config: &CameraConfig) {
    common::setup_logging(config.environment.clone());
}
