use crate::{config::ControllerConfig, mqtt_notifier::MqttNotifier, state_machine::StateContext};
use anyhow::Result;
use bridge::{BridgeSemaphore, DetectionReader, SemaphoreType, SentryControl};
use common::wait_for_resource;
use std::{thread, time::Duration};

pub struct ControllerService {
    config: ControllerConfig,
    state_context: StateContext,
    detection_reader: DetectionReader,
    detection_semaphore: BridgeSemaphore,
    mode_semaphore: BridgeSemaphore,
    sentry_control: SentryControl,
    mqtt_notifier: MqttNotifier,
}

impl ControllerService {
    pub fn new(config: ControllerConfig) -> Result<Self> {
        let detection_reader = wait_for_resource(
            DetectionReader::build,
            config.poll_interval_ms,
            "Detection buffer",
        );

        let detection_semaphore = wait_for_resource(
            || BridgeSemaphore::open(SemaphoreType::DetectionInferenceToController),
            config.poll_interval_ms,
            "Detection semaphore",
        );

        let mode_semaphore = BridgeSemaphore::ensure(SemaphoreType::ModeChangeControllerToCapture)
            .map_err(|e| anyhow::anyhow!("Failed to create mode change semaphore: {}", e))?;
        tracing::info!("Mode change semaphore connected");

        let sentry_control = SentryControl::build()?;
        tracing::info!("Sentry control connected");

        let mqtt_notifier = MqttNotifier::new(
            &config.mqtt_broker_host,
            config.mqtt_broker_port,
            config.mqtt_topic.clone(),
            config.mqtt_device_id.clone(),
        )?;

        Ok(Self {
            config,
            state_context: StateContext::new(),
            detection_reader,
            detection_semaphore,
            mode_semaphore,
            sentry_control,
            mqtt_notifier,
        })
    }

    pub fn run(mut self) -> Result<()> {
        tracing::info!("Controller service starting");
        tracing::info!(
            "State machine config - Validation frames: {}, Tracking exit frames: {}",
            self.config.validation_frames,
            self.config.tracking_exit_frames
        );

        let mut frames_processed = 0u64;

        loop {
            if let Err(e) = self.detection_semaphore.wait() {
                tracing::error!(error = %e, "Semaphore wait failed");
                thread::sleep(Duration::from_millis(self.config.poll_interval_ms));
                continue;
            }

            let person_detected = match self.detection_reader.check_person_detected() {
                Ok(detected) => detected,
                Err(e) => {
                    tracing::error!(error = %e, "Failed to read detections");
                    continue;
                }
            };

            let previous_state = self.state_context.current_state();

            let state_changed = self.state_context.update(
                person_detected,
                self.config.validation_frames,
                self.config.tracking_exit_frames,
            );

            if let Some(new_state) = state_changed {
                let sentry_mode = self.state_context.to_sentry_mode();
                self.sentry_control.set_mode(sentry_mode);

                // Signal capture to wake up immediately for mode change
                if let Err(e) = self.mode_semaphore.post() {
                    tracing::warn!(error = %e, "Failed to signal mode change to capture");
                }

                tracing::info!(
                    state = ?new_state,
                    sentry_mode = ?sentry_mode,
                    "State transition"
                );

                // Send MQTT notifications only for:
                // 1. Entering Tracking state (human presence validated)
                // 2. Tracking -> Standby transition (human left)
                use crate::state_machine::ControllerState;
                let should_notify = matches!(new_state, ControllerState::Tracking)
                    || (matches!(new_state, ControllerState::Standby)
                        && matches!(previous_state, ControllerState::Tracking));

                if should_notify
                    && let Err(e) = self
                        .mqtt_notifier
                        .notify_state_change(new_state, Some(previous_state))
                {
                    tracing::error!(error = %e, "Failed to send MQTT notification");
                }
            }

            frames_processed += 1;
            if frames_processed.is_multiple_of(30) {
                tracing::debug!(
                    frames_processed,
                    current_state = ?self.state_context.current_state(),
                    person_detected,
                    "Controller status"
                );
            }

            self.detection_reader.mark_read();
        }
    }
}
