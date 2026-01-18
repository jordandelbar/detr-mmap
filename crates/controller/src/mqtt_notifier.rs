use anyhow::{Context, Result};
use chrono::Utc;
use rumqttc::{Client, ConnectionError, Event, MqttOptions, Packet, QoS};
use serde::Serialize;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
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

#[allow(dead_code)]
pub struct MqttNotifier {
    client: Client,
    topic: String,
    device_id: String,
    connected: Arc<AtomicBool>,
}

impl MqttNotifier {
    pub fn new(
        broker_host: &str,
        broker_port: u16,
        topic: String,
        device_id: String,
    ) -> Result<Self> {
        let mut mqtt_options = MqttOptions::new("detr-mmap-controller", broker_host, broker_port);
        mqtt_options.set_keep_alive(Duration::from_secs(30));
        mqtt_options.set_clean_session(true);

        let (client, mut connection) = Client::new(mqtt_options, 10);
        let connected = Arc::new(AtomicBool::new(false));
        let connected_clone = Arc::clone(&connected);

        std::thread::spawn(move || {
            let mut reconnect_attempts = 0u32;

            loop {
                for notification in connection.iter() {
                    match notification {
                        Ok(Event::Incoming(Packet::ConnAck(_))) => {
                            connected_clone.store(true, Ordering::Release);
                            reconnect_attempts = 0;
                            tracing::info!("MQTT connected to broker");
                        }
                        Ok(Event::Incoming(Packet::PingResp)) => {
                            tracing::trace!("MQTT ping response received");
                        }
                        Ok(_) => {}
                        Err(e) => {
                            connected_clone.store(false, Ordering::Release);
                            match &e {
                                ConnectionError::Io(_) | ConnectionError::NetworkTimeout => {
                                    reconnect_attempts = reconnect_attempts.saturating_add(1);
                                    let backoff = calculate_backoff(reconnect_attempts);
                                    tracing::warn!(
                                        error = %e,
                                        attempt = reconnect_attempts,
                                        backoff_ms = backoff.as_millis(),
                                        "MQTT connection lost, reconnecting"
                                    );
                                    std::thread::sleep(backoff);
                                }
                                _ => {
                                    tracing::error!(error = %e, "MQTT error");
                                }
                            }
                        }
                    }
                }

                // Connection iterator ended - this happens on disconnect
                // rumqttc will automatically try to reconnect when we iterate again
                connected_clone.store(false, Ordering::Release);
                reconnect_attempts = reconnect_attempts.saturating_add(1);
                let backoff = calculate_backoff(reconnect_attempts);
                tracing::warn!(
                    attempt = reconnect_attempts,
                    backoff_ms = backoff.as_millis(),
                    "MQTT connection closed, attempting reconnect"
                );
                std::thread::sleep(backoff);
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
            connected,
        })
    }

    /// Returns true if currently connected to the MQTT broker
    #[allow(dead_code)]
    pub fn is_connected(&self) -> bool {
        self.connected.load(Ordering::Acquire)
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

/// Calculate exponential backoff with jitter, capped at 30 seconds
fn calculate_backoff(attempt: u32) -> Duration {
    const BASE_MS: u64 = 100;
    const MAX_MS: u64 = 30_000;

    let exp_backoff = BASE_MS.saturating_mul(2u64.saturating_pow(attempt.min(10)));
    let capped = exp_backoff.min(MAX_MS);

    let jitter = (capped / 10).max(1);
    let jittered = capped.saturating_add(fastrand::u64(0..jitter));

    Duration::from_millis(jittered)
}
