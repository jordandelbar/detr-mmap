use crate::errors::BridgeError;
use crate::paths;
use memmap2::MmapMut;
use std::fs::OpenOptions;
use std::os::unix::fs::OpenOptionsExt;
use std::sync::atomic::{AtomicU8, Ordering};

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SentryMode {
    Standby = 0,
    Alarmed = 1,
}

impl SentryMode {
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(SentryMode::Standby),
            1 => Some(SentryMode::Alarmed),
            _ => None,
        }
    }
}

pub struct SentryControl {
    _mmap: MmapMut,
    mode: &'static AtomicU8,
}

unsafe impl Send for SentryControl {}
unsafe impl Sync for SentryControl {}

impl SentryControl {
    /// Create a new SentryControl using the default sentry control path
    pub fn build() -> Result<Self, BridgeError> {
        Self::new(paths::SENTRY_CONTROL_PATH)
    }

    /// Create or open shared memory control with custom path (useful for tests)
    ///
    /// This creates a shared memory segment in /dev/shm for the sentry mode.
    /// The segment is 1 byte containing an atomic U8.
    ///
    /// # Arguments
    /// * `path` - Path in /dev/shm (e.g., "/dev/shm/bridge_sentry_control")
    pub fn new(path: &str) -> Result<Self, BridgeError> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .mode(0o600)
            .open(path)?;

        let metadata = file.metadata()?;

        if metadata.len() == 0 {
            file.set_len(1)?;
        }

        let mut mmap = unsafe { MmapMut::map_mut(&file)? };

        let ptr = mmap.as_mut_ptr() as *const AtomicU8;
        let mode = unsafe { &*ptr };

        Ok(Self { _mmap: mmap, mode })
    }

    #[inline]
    pub fn get_mode(&self) -> SentryMode {
        let value = self.mode.load(Ordering::Acquire);
        SentryMode::from_u8(value).unwrap_or(SentryMode::Standby)
    }

    #[inline]
    pub fn set_mode(&self, mode: SentryMode) {
        self.mode.store(mode as u8, Ordering::Release);
    }

    #[inline]
    pub fn try_set_mode(&self, mode: SentryMode) -> bool {
        let current = self.mode.load(Ordering::Acquire);
        if current == mode as u8 {
            return false;
        }
        self.mode.store(mode as u8, Ordering::Release);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sentry_control_create_and_set() {
        let path = "/dev/shm/test_sentry_control";
        let _ = std::fs::remove_file(path); // Clean up from previous runs

        let control = SentryControl::new(path).expect("Failed to create control");

        // Default should be Standby
        assert_eq!(control.get_mode(), SentryMode::Standby);

        // Set to Alarmed
        control.set_mode(SentryMode::Alarmed);
        assert_eq!(control.get_mode(), SentryMode::Alarmed);

        // Set back to Standby
        control.set_mode(SentryMode::Standby);
        assert_eq!(control.get_mode(), SentryMode::Standby);

        // Cleanup
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_sentry_control_shared_across_instances() {
        let path = "/dev/shm/test_sentry_control_shared";
        let _ = std::fs::remove_file(path);

        let control1 = SentryControl::new(path).expect("Failed to create control1");
        let control2 = SentryControl::new(path).expect("Failed to create control2");

        // Write with control1
        control1.set_mode(SentryMode::Alarmed);

        // Read with control2
        assert_eq!(control2.get_mode(), SentryMode::Alarmed);

        // Write with control2
        control2.set_mode(SentryMode::Standby);

        // Read with control1
        assert_eq!(control1.get_mode(), SentryMode::Standby);

        // Cleanup
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_try_set_mode() {
        let path = "/dev/shm/test_sentry_try_set";
        let _ = std::fs::remove_file(path);

        let control = SentryControl::new(path).expect("Failed to create control");

        // First set should return true
        assert!(control.try_set_mode(SentryMode::Alarmed));

        // Setting same mode should return false
        assert!(!control.try_set_mode(SentryMode::Alarmed));

        // Setting different mode should return true
        assert!(control.try_set_mode(SentryMode::Standby));

        // Cleanup
        let _ = std::fs::remove_file(path);
    }
}
