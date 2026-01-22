use bridge::SentryMode;
use std::time::Duration;

pub struct CapturePacing {
    standby: Duration,
    alarmed: Duration,
    current: Duration,
    mode: SentryMode,
}

impl CapturePacing {
    pub fn new(max_fps: f64, sentry_fps: f64) -> Self {
        let standby = Duration::from_secs_f64(1.0 / sentry_fps);
        let alarmed = Duration::from_secs_f64(1.0 / max_fps);

        Self {
            standby,
            alarmed,
            current: standby,
            mode: SentryMode::Standby,
        }
    }

    /// Update pacing for new mode. Returns true if mode changed.
    pub fn update(&mut self, new_mode: SentryMode) -> bool {
        if new_mode == self.mode {
            return false;
        }

        self.mode = new_mode;
        self.current = match new_mode {
            SentryMode::Standby => self.standby,
            SentryMode::Alarmed => self.alarmed,
        };
        true
    }

    pub fn mode(&self) -> SentryMode {
        self.mode
    }

    pub fn frame_duration(&self) -> Duration {
        self.current
    }
}
