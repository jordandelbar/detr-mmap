use anyhow::Result;
use std::env;

#[derive(Debug, Clone)]
pub struct ControllerConfig {
    pub detection_mmap_path: String,
    pub controller_semaphore_name: String,
    pub sentry_control_path: String,
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
        let detection_mmap_path = env::var("DETECTION_MMAP_PATH")
            .unwrap_or_else(|_| "/dev/shm/bridge_detection_buffer".to_string());

        let controller_semaphore_name = env::var("CONTROLLER_SEMAPHORE_NAME")
            .unwrap_or_else(|_| "/bridge_detection_controller".to_string());

        let sentry_control_path = env::var("SENTRY_CONTROL_PATH")
            .unwrap_or_else(|_| "/dev/shm/bridge_sentry_control".to_string());

        let validation_frames = env::var("VALIDATION_FRAMES")
            .unwrap_or_else(|_| "3".to_string())
            .parse()
            .unwrap_or(3);

        let tracking_exit_frames = env::var("TRACKING_EXIT_FRAMES")
            .unwrap_or_else(|_| "10".to_string())
            .parse()
            .unwrap_or(10);

        let poll_interval_ms = env::var("POLL_INTERVAL_MS")
            .unwrap_or_else(|_| "500".to_string())
            .parse()
            .unwrap_or(500);

        let mqtt_broker_host = env::var("MQTT_BROKER_HOST")
            .unwrap_or_else(|_| "mosquitto".to_string());

        let mqtt_broker_port = env::var("MQTT_BROKER_PORT")
            .unwrap_or_else(|_| "1883".to_string())
            .parse()
            .unwrap_or(1883);

        let mqtt_topic = env::var("MQTT_TOPIC")
            .unwrap_or_else(|_| "bridge-rt/controller/state".to_string());

        let mqtt_device_id = env::var("MQTT_DEVICE_ID")
            .unwrap_or_else(|_| "unknown".to_string());

        Ok(Self {
            detection_mmap_path,
            controller_semaphore_name,
            sentry_control_path,
            validation_frames,
            tracking_exit_frames,
            poll_interval_ms,
            mqtt_broker_host,
            mqtt_broker_port,
            mqtt_topic,
            mqtt_device_id,
        })
    }
}
