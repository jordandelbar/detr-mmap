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

#[cfg(test)]
mod tests {
    use super::*;

    // ========== Initial State Tests ==========

    #[test]
    fn new_starts_in_standby() {
        let ctx = StateContext::new();
        assert_eq!(ctx.current_state(), ControllerState::Standby);
    }

    #[test]
    fn new_initializes_counters_to_zero() {
        let ctx = StateContext::new();
        assert_eq!(ctx.validation_count, 0);
        assert_eq!(ctx.no_person_count, 0);
    }

    // ========== Standby State Transitions ==========

    #[test]
    fn standby_no_person_stays_standby() {
        let mut ctx = StateContext::new();
        let result = ctx.update(false, 3, 5);

        assert_eq!(ctx.current_state(), ControllerState::Standby);
        assert!(result.is_none(), "No state change should return None");
    }

    #[test]
    fn standby_person_detected_transitions_to_validation() {
        let mut ctx = StateContext::new();
        let result = ctx.update(true, 3, 5);

        assert_eq!(ctx.current_state(), ControllerState::Validation);
        assert_eq!(result, Some(ControllerState::Validation));
        assert_eq!(ctx.validation_count, 1);
    }

    // ========== Validation State Transitions ==========

    #[test]
    fn validation_person_detected_increments_count() {
        let mut ctx = StateContext::new();
        ctx.update(true, 5, 5); // Enter Validation

        let result = ctx.update(true, 5, 5);

        assert_eq!(ctx.current_state(), ControllerState::Validation);
        assert!(result.is_none(), "Should stay in Validation");
        assert_eq!(ctx.validation_count, 2);
    }

    #[test]
    fn validation_reaches_threshold_transitions_to_tracking() {
        let mut ctx = StateContext::new();
        let threshold = 3;

        // First detection: Standby -> Validation (count = 1)
        ctx.update(true, threshold, 5);
        assert_eq!(ctx.validation_count, 1);

        // Second detection: stay Validation (count = 2)
        ctx.update(true, threshold, 5);
        assert_eq!(ctx.validation_count, 2);

        // Third detection: count reaches threshold -> Tracking
        let result = ctx.update(true, threshold, 5);

        assert_eq!(ctx.current_state(), ControllerState::Tracking);
        assert_eq!(result, Some(ControllerState::Tracking));
    }

    #[test]
    fn validation_no_person_returns_to_standby() {
        let mut ctx = StateContext::new();
        ctx.update(true, 3, 5); // Enter Validation
        assert_eq!(ctx.current_state(), ControllerState::Validation);

        let result = ctx.update(false, 3, 5);

        assert_eq!(ctx.current_state(), ControllerState::Standby);
        assert_eq!(result, Some(ControllerState::Standby));
        assert_eq!(ctx.validation_count, 0, "Count should reset");
    }

    // ========== Tracking State Transitions ==========

    #[test]
    fn tracking_person_detected_stays_tracking() {
        let mut ctx = StateContext::new();
        // Get to Tracking state
        ctx.update(true, 1, 5); // Standby -> Validation
        ctx.update(true, 1, 5); // Validation -> Tracking

        let result = ctx.update(true, 1, 5);

        assert_eq!(ctx.current_state(), ControllerState::Tracking);
        assert!(result.is_none());
        assert_eq!(ctx.no_person_count, 0);
    }

    #[test]
    fn tracking_no_person_increments_no_person_count() {
        let mut ctx = StateContext::new();
        ctx.update(true, 1, 5);
        ctx.update(true, 1, 5); // Now in Tracking

        let result = ctx.update(false, 1, 5);

        assert_eq!(ctx.current_state(), ControllerState::Tracking);
        assert!(result.is_none());
        assert_eq!(ctx.no_person_count, 1);
    }

    #[test]
    fn tracking_no_person_reaches_threshold_returns_to_standby() {
        let mut ctx = StateContext::new();
        let exit_threshold = 3;

        ctx.update(true, 1, exit_threshold);
        ctx.update(true, 1, exit_threshold); // Now in Tracking

        // Miss 1
        ctx.update(false, 1, exit_threshold);
        assert_eq!(ctx.no_person_count, 1);

        // Miss 2
        ctx.update(false, 1, exit_threshold);
        assert_eq!(ctx.no_person_count, 2);

        // Miss 3 - should exit
        let result = ctx.update(false, 1, exit_threshold);

        assert_eq!(ctx.current_state(), ControllerState::Standby);
        assert_eq!(result, Some(ControllerState::Standby));
        assert_eq!(ctx.validation_count, 0);
        assert_eq!(ctx.no_person_count, 0);
    }

    #[test]
    fn tracking_person_resets_no_person_count() {
        let mut ctx = StateContext::new();
        ctx.update(true, 1, 5);
        ctx.update(true, 1, 5); // Now in Tracking

        // Accumulate some no-person counts
        ctx.update(false, 1, 5);
        ctx.update(false, 1, 5);
        assert_eq!(ctx.no_person_count, 2);

        // Person detected again - should reset counter
        ctx.update(true, 1, 5);

        assert_eq!(ctx.current_state(), ControllerState::Tracking);
        assert_eq!(ctx.no_person_count, 0);
    }

    // ========== Edge Cases ==========

    #[test]
    fn flaky_detection_in_validation_resets() {
        let mut ctx = StateContext::new();

        // Start validating
        ctx.update(true, 5, 5);
        ctx.update(true, 5, 5);
        assert_eq!(ctx.validation_count, 2);

        // Flaky: lose detection
        ctx.update(false, 5, 5);
        assert_eq!(ctx.current_state(), ControllerState::Standby);
        assert_eq!(ctx.validation_count, 0);

        // Start over
        ctx.update(true, 5, 5);
        assert_eq!(ctx.current_state(), ControllerState::Validation);
        assert_eq!(ctx.validation_count, 1);
    }

    #[test]
    fn full_cycle_standby_to_tracking_and_back() {
        let mut ctx = StateContext::new();
        let validation_threshold = 2;
        let exit_threshold = 2;

        // Standby -> Validation
        assert_eq!(
            ctx.update(true, validation_threshold, exit_threshold),
            Some(ControllerState::Validation)
        );

        // Validation -> Tracking
        assert_eq!(
            ctx.update(true, validation_threshold, exit_threshold),
            Some(ControllerState::Tracking)
        );

        // Stay Tracking (person still there)
        assert!(
            ctx.update(true, validation_threshold, exit_threshold)
                .is_none()
        );

        // Start losing person
        assert!(
            ctx.update(false, validation_threshold, exit_threshold)
                .is_none()
        );

        // Tracking -> Standby
        assert_eq!(
            ctx.update(false, validation_threshold, exit_threshold),
            Some(ControllerState::Standby)
        );

        // Stay Standby
        assert!(
            ctx.update(false, validation_threshold, exit_threshold)
                .is_none()
        );
    }

    #[test]
    fn threshold_boundary_validation() {
        let mut ctx = StateContext::new();
        let threshold = 3;

        ctx.update(true, threshold, 5); // count = 1
        ctx.update(true, threshold, 5); // count = 2
        assert_eq!(ctx.current_state(), ControllerState::Validation);

        // count = 3, which equals threshold
        ctx.update(true, threshold, 5);
        assert_eq!(ctx.current_state(), ControllerState::Tracking);
    }

    #[test]
    fn threshold_boundary_tracking_exit() {
        let mut ctx = StateContext::new();
        let exit_threshold = 2;

        // Get to Tracking
        ctx.update(true, 1, exit_threshold);
        ctx.update(true, 1, exit_threshold);
        assert_eq!(ctx.current_state(), ControllerState::Tracking);

        // count = 1, below threshold
        ctx.update(false, 1, exit_threshold);
        assert_eq!(ctx.current_state(), ControllerState::Tracking);

        // count = 2, equals threshold - should exit
        ctx.update(false, 1, exit_threshold);
        assert_eq!(ctx.current_state(), ControllerState::Standby);
    }

    // ========== to_sentry_mode Tests ==========

    #[test]
    fn sentry_mode_standby() {
        let ctx = StateContext::new();
        assert_eq!(ctx.to_sentry_mode(), SentryMode::Standby);
    }

    #[test]
    fn sentry_mode_validation_is_alarmed() {
        let mut ctx = StateContext::new();
        ctx.update(true, 5, 5); // Enter Validation

        assert_eq!(ctx.to_sentry_mode(), SentryMode::Alarmed);
    }

    #[test]
    fn sentry_mode_tracking_is_alarmed() {
        let mut ctx = StateContext::new();
        ctx.update(true, 1, 5);
        ctx.update(true, 1, 5); // Enter Tracking

        assert_eq!(ctx.to_sentry_mode(), SentryMode::Alarmed);
    }
}
