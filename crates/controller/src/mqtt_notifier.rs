use anyhow::{Context, Result};
use chrono::Utc;
use rumqttc::{Client, MqttOptions, QoS};
use serde::Serialize;
use std::time::Duration;

use crate::state_machine::ControllerState;

#[derive(Debug, Serialize)]
pub struct StateChangeNotification {
    pub device_id: String,
    pub timestamp: String,
    pub state: String,
    pub previous_state: Option<String>,
    pub event_type: String,
}

pub struct MqttNotifier {
    client: Client,
    topic: String,
    device_id: String,
}

impl MqttNotifier {
    pub fn new(
        broker_host: &str,
        broker_port: u16,
        topic: String,
        device_id: String,
    ) -> Result<Self> {
        let mut mqtt_options = MqttOptions::new("bridge-rt-controller", broker_host, broker_port);
        mqtt_options.set_keep_alive(Duration::from_secs(30));

        let (client, mut connection) = Client::new(mqtt_options, 10);

        std::thread::spawn(move || {
            for notification in connection.iter() {
                match notification {
                    Ok(_) => {}
                    Err(e) => {
                        tracing::error!(error = %e, "MQTT connection error");
                    }
                }
            }
        });

        tracing::info!(
            broker = %format!("{}:{}", broker_host, broker_port),
            topic = %topic,
            device_id = %device_id,
            "MQTT notifier initialized"
        );

        Ok(Self {
            client,
            topic,
            device_id,
        })
    }

    pub fn notify_state_change(
        &self,
        new_state: ControllerState,
        previous_state: Option<ControllerState>,
    ) -> Result<()> {
        let event_type = match new_state {
            ControllerState::Tracking => "human_detected",
            ControllerState::Standby => "standby_resumed",
            ControllerState::Validation => "validation_started",
        };

        let notification = StateChangeNotification {
            device_id: self.device_id.clone(),
            timestamp: Utc::now().to_rfc3339(),
            state: format!("{:?}", new_state),
            previous_state: previous_state.map(|s| format!("{:?}", s)),
            event_type: event_type.to_string(),
        };

        let payload = serde_json::to_string(&notification)
            .context("Failed to serialize state change notification")?;

        self.client
            .publish(&self.topic, QoS::AtLeastOnce, false, payload.as_bytes())
            .context("Failed to publish MQTT message")?;

        tracing::debug!(
            state = ?new_state,
            event_type = %event_type,
            "State change notification published"
        );

        Ok(())
    }
}
