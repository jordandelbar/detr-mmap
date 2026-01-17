use anyhow::Result;
use common::get_env;

#[derive(Debug, Clone)]
pub struct ControllerConfig {
    pub validation_frames: u32,
    pub tracking_exit_frames: u32,
    pub poll_interval_ms: u64,
    pub mqtt_broker_host: String,
    pub mqtt_broker_port: u16,
    pub mqtt_topic: String,
    pub mqtt_device_id: String,
}

impl ControllerConfig {
    pub fn from_env() -> Result<Self> {
        Ok(Self {
            validation_frames: get_env("VALIDATION_FRAMES", 3),
            tracking_exit_frames: get_env("TRACKING_EXIT_FRAMES", 10),
            poll_interval_ms: get_env("POLL_INTERVAL_MS", 500),
            mqtt_broker_host: get_env("MQTT_BROKER_HOST", "mosquitto".to_string()),
            mqtt_broker_port: get_env("MQTT_BROKER_PORT", 1883),
            mqtt_topic: get_env("MQTT_TOPIC", "detr-mmap/controller/state".to_string()),
            mqtt_device_id: get_env("MQTT_DEVICE_ID", "unknown".to_string()),
        })
    }
}
