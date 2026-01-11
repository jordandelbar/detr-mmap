//! Centralized IPC path configuration
//!
//! This module defines all shared memory paths, semaphore names, and buffer sizes
//! used for inter-process communication in the bridge system.
//!
//! Having these in one place ensures:
//! - No path mismatches between producers and consumers
//! - Single source of truth for IPC configuration

/// Frame buffer path - used by capture (write) and inference + gateway (read)
pub const FRAME_BUFFER_PATH: &str = "/dev/shm/bridge_frame_buffer";

/// Detection buffer path - used by inference (write) and gateway + controller (read)
pub const DETECTION_BUFFER_PATH: &str = "/dev/shm/bridge_detection_buffer";

/// Sentry control shared memory path - used by controller (write) and capture (read)
pub const SENTRY_CONTROL_PATH: &str = "/dev/shm/bridge_sentry_control";

/// Semaphore name for inference frame synchronization
pub const SEMAPHORE_FRAME_INFERENCE: &str = "/bridge_frame_inference";

/// Semaphore name for gateway frame synchronization
pub const SEMAPHORE_FRAME_GATEWAY: &str = "/bridge_frame_gateway";

/// Semaphore name for controller detection notifications
pub const SEMAPHORE_DETECTION_CONTROLLER: &str = "/bridge_detection_controller";

/// Default frame buffer size (6MB - enough for 1920x1080 RGB)
pub const DEFAULT_FRAME_BUFFER_SIZE: usize = 6 * 1024 * 1024;

/// Default detection buffer size (1MB - enough for many detections)
pub const DEFAULT_DETECTION_BUFFER_SIZE: usize = 1024 * 1024;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_paths_are_absolute() {
        assert!(FRAME_BUFFER_PATH.starts_with('/'));
        assert!(DETECTION_BUFFER_PATH.starts_with('/'));
        assert!(SENTRY_CONTROL_PATH.starts_with('/'));
    }

    #[test]
    fn test_semaphore_names_start_with_slash() {
        assert!(SEMAPHORE_FRAME_INFERENCE.starts_with('/'));
        assert!(SEMAPHORE_FRAME_GATEWAY.starts_with('/'));
        assert!(SEMAPHORE_DETECTION_CONTROLLER.starts_with('/'));
    }

    #[test]
    fn test_buffer_sizes_reasonable() {
        assert!(DEFAULT_FRAME_BUFFER_SIZE >= 1024 * 1024); // At least 1MB
        assert!(DEFAULT_DETECTION_BUFFER_SIZE >= 1024); // At least 1KB
    }
}
