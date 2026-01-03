use bridge::SentryMode;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ControllerState {
    Standby,
    Validation,
    Tracking,
}

pub struct StateContext {
    current_state: ControllerState,
    validation_count: u32,
    no_person_count: u32,
}

impl StateContext {
    pub fn new() -> Self {
        Self {
            current_state: ControllerState::Standby,
            validation_count: 0,
            no_person_count: 0,
        }
    }

    pub fn update(
        &mut self,
        person_detected: bool,
        validation_threshold: u32,
        tracking_exit_threshold: u32,
    ) -> Option<ControllerState> {
        let old_state = self.current_state;

        match self.current_state {
            ControllerState::Standby => {
                if person_detected {
                    self.current_state = ControllerState::Validation;
                    self.validation_count = 1;
                    self.no_person_count = 0;
                }
            }
            ControllerState::Validation => {
                if person_detected {
                    self.validation_count += 1;
                    if self.validation_count >= validation_threshold {
                        self.current_state = ControllerState::Tracking;
                        self.no_person_count = 0;
                    }
                } else {
                    self.current_state = ControllerState::Standby;
                    self.validation_count = 0;
                }
            }
            ControllerState::Tracking => {
                if person_detected {
                    self.no_person_count = 0;
                } else {
                    self.no_person_count += 1;
                    if self.no_person_count >= tracking_exit_threshold {
                        self.current_state = ControllerState::Standby;
                        self.validation_count = 0;
                        self.no_person_count = 0;
                    }
                }
            }
        }

        if old_state != self.current_state {
            Some(self.current_state)
        } else {
            None
        }
    }

    pub fn current_state(&self) -> ControllerState {
        self.current_state
    }

    pub fn to_sentry_mode(&self) -> SentryMode {
        match self.current_state {
            ControllerState::Standby => SentryMode::Standby,
            ControllerState::Validation | ControllerState::Tracking => SentryMode::Alarmed,
        }
    }
}
