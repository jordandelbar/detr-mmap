use crate::errors::BridgeError;
use nix::mqueue::{
    MQ_OFlag, MqAttr, MqdT, mq_close, mq_open, mq_receive, mq_send, mq_timedreceive, mq_unlink,
};
use nix::sys::stat::Mode;
use nix::sys::time::TimeSpec;
use std::ffi::CString;

#[derive(Copy, Clone)]
pub enum SemaphoreType {
    FrameCaptureToInference,
    FrameCaptureToGateway,
    DetectionInferenceToController,
}

impl SemaphoreType {
    fn name(&self) -> &str {
        match self {
            Self::FrameCaptureToInference => "/bridge_frame_inference",
            Self::FrameCaptureToGateway => "/bridge_frame_gateway",
            Self::DetectionInferenceToController => "/bridge_detection_controller",
        }
    }
}

/// A wrapper around POSIX message queues for frame synchronization
///
/// This uses message queues to signal when new frames are available.
/// The gateway posts to the queue after writing each frame,
/// signaling both inference and gateway processes.
///
/// Note: Message queues are treated as persistent system resources.
/// They are not automatically deleted when this struct is dropped,
/// allowing seamless pod restarts in Kubernetes.
pub struct BridgeSemaphore {
    mqd: Option<MqdT>,
}

unsafe impl Send for BridgeSemaphore {}
unsafe impl Sync for BridgeSemaphore {}

impl BridgeSemaphore {
    pub fn create(semaphore_type: SemaphoreType) -> Result<Self, BridgeError> {
        Self::create_with_name(semaphore_type.name())
    }

    pub fn open(semaphore_type: SemaphoreType) -> Result<Self, BridgeError> {
        Self::open_with_name(semaphore_type.name())
    }
    /// Create a new message queue
    ///
    /// This will create a new message queue or open an existing one.
    /// The queue can hold up to 10 messages (signals).
    ///
    /// # Arguments
    /// * `name` - Name of the message queue (e.g., "/bridge_frame_ready")
    ///
    /// # Returns
    /// A new BridgeSemaphore instance that owns the message queue
    pub fn create_with_name(name: &str) -> Result<Self, BridgeError> {
        let c_name = CString::new(name)
            .map_err(|e| BridgeError::SemaphoreError(format!("Invalid queue name: {}", e)))?;

        // Try to unlink any existing queue first
        let _ = mq_unlink(c_name.as_c_str());

        // Set queue attributes: max 10 messages, 1 byte per message
        let attr = MqAttr::new(0, 10, 1, 0);

        // Create new message queue
        let mqd = mq_open(
            c_name.as_c_str(),
            MQ_OFlag::O_CREAT | MQ_OFlag::O_EXCL | MQ_OFlag::O_RDWR,
            Mode::S_IRUSR | Mode::S_IWUSR | Mode::S_IRGRP | Mode::S_IWGRP,
            Some(&attr),
        )
        .map_err(|e| BridgeError::SemaphoreError(format!("Failed to create queue: {}", e)))?;

        Ok(Self { mqd: Some(mqd) })
    }

    /// Open an existing message queue
    ///
    /// # Arguments
    /// * `name` - Name of the message queue (e.g., "/bridge_frame_ready")
    ///
    /// # Returns
    /// A new BridgeSemaphore instance connected to the existing queue
    pub fn open_with_name(name: &str) -> Result<Self, BridgeError> {
        let c_name = CString::new(name)
            .map_err(|e| BridgeError::SemaphoreError(format!("Invalid queue name: {}", e)))?;

        // Open existing message queue
        let mqd = mq_open(c_name.as_c_str(), MQ_OFlag::O_RDWR, Mode::empty(), None)
            .map_err(|e| BridgeError::SemaphoreError(format!("Failed to open queue: {}", e)))?;

        Ok(Self { mqd: Some(mqd) })
    }

    /// Wait for a signal
    ///
    /// This will block until a message (signal) is available in the queue.
    /// Automatically retries if interrupted by signals.
    pub fn wait(&self) -> Result<(), BridgeError> {
        let mut buf = [0u8; 1];
        let mut prio = 0u32;
        let mqd = self.mqd.as_ref().expect("Message queue descriptor is None");

        loop {
            match mq_receive(mqd, &mut buf, &mut prio) {
                Ok(_) => return Ok(()),
                Err(nix::errno::Errno::EINTR) => continue, // Retry on interrupt
                Err(e) => {
                    return Err(BridgeError::SemaphoreError(format!(
                        "Queue receive failed: {}",
                        e
                    )));
                }
            }
        }
    }

    /// Wait for a signal with a timeout
    ///
    /// This will block until a message (signal) is available or the timeout expires.
    /// Returns Ok(true) if a signal was received, Ok(false) if timeout occurred.
    /// Automatically retries if interrupted by signals.
    pub fn wait_timeout(&self, timeout_secs: u64) -> Result<bool, BridgeError> {
        let mut buf = [0u8; 1];
        let mut prio = 0u32;
        let mqd = self.mqd.as_ref().expect("Message queue descriptor is None");
        let timeout = TimeSpec::new(timeout_secs as i64, 0);

        loop {
            match mq_timedreceive(mqd, &mut buf, &mut prio, &timeout) {
                Ok(_) => return Ok(true),
                Err(nix::errno::Errno::EINTR) => continue, // Retry on interrupt
                Err(nix::errno::Errno::ETIMEDOUT) => return Ok(false), // Timeout
                Err(e) => {
                    return Err(BridgeError::SemaphoreError(format!(
                        "Queue timed receive failed: {}",
                        e
                    )));
                }
            }
        }
    }

    /// Try to wait without blocking
    ///
    /// Returns Ok(true) if a signal was consumed, Ok(false) if none available.
    /// This is used by inference to "drain" pending signals and skip to the latest frame.
    pub fn try_wait(&self) -> Result<bool, BridgeError> {
        let mut buf = [0u8; 1];
        let mut prio = 0u32;
        let mqd = self.mqd.as_ref().expect("Message queue descriptor is None");

        // Use timed receive with zero timeout for non-blocking behavior
        let timeout = TimeSpec::new(0, 0);

        match mq_timedreceive(mqd, &mut buf, &mut prio, &timeout) {
            Ok(_) => Ok(true),
            Err(nix::errno::Errno::ETIMEDOUT) => Ok(false), // No message available
            Err(e) => Err(BridgeError::SemaphoreError(format!(
                "Queue try_receive failed: {}",
                e
            ))),
        }
    }

    /// Signal the queue (send a message)
    ///
    /// Gateway calls this after writing a frame.
    /// It should be called twice per frame (once for inference, once for gateway)
    /// to implement the fan-out pattern.
    pub fn post(&self) -> Result<(), BridgeError> {
        let msg = [1u8]; // Simple 1-byte message
        let mqd = self.mqd.as_ref().expect("Message queue descriptor is None");
        mq_send(mqd, &msg, 0)
            .map_err(|e| BridgeError::SemaphoreError(format!("Queue send failed: {}", e)))
    }

    /// Drain all pending signals
    ///
    /// Used by inference to skip all queued signals and process only the latest frame.
    /// Returns the number of signals drained.
    pub fn drain(&self) -> Result<usize, BridgeError> {
        let mut count = 0;
        while self.try_wait()? {
            count += 1;
        }
        Ok(count)
    }
}

impl Drop for BridgeSemaphore {
    fn drop(&mut self) {
        // Close the message queue descriptor
        if let Some(mqd) = self.mqd.take() {
            let _ = mq_close(mqd);
        }

        // NOTE: We intentionally DO NOT unlink the queue here.
        // Message queues are treated as persistent system resources that survive pod restarts.
        // This allows:
        // - Capture pod can restart without breaking gateway/inference listeners
        // - Gateway/inference keep their file descriptors and continue receiving messages
        // - No need for complex reconnection logic
        //
        // Cleanup should happen via:
        // - Init containers in Kubernetes
        // - Manual cleanup during system maintenance
        // - Or automatic cleanup on host reboot (queues are in /dev/mqueue)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mqueue_create_and_open() {
        let queue_name = "/test_bridge_queue1";

        // Create queue
        let creator =
            BridgeSemaphore::create_with_name(queue_name).expect("Failed to create queue");

        // Open existing queue
        let opener = BridgeSemaphore::open_with_name(queue_name).expect("Failed to open queue");

        // Post from creator
        creator.post().expect("Failed to post");

        // Wait from opener
        opener.wait().expect("Failed to wait");

        // Cleanup happens automatically via Drop
    }

    #[test]
    fn test_mqueue_drain() {
        let queue_name = "/test_bridge_queue2";

        let mq = BridgeSemaphore::create_with_name(queue_name).expect("Failed to create queue");

        // Post 5 times
        for _ in 0..5 {
            mq.post().expect("Failed to post");
        }

        // Drain all signals
        let drained = mq.drain().expect("Failed to drain");
        assert_eq!(drained, 5);

        // No more signals available
        let drained = mq.drain().expect("Failed to drain");
        assert_eq!(drained, 0);
    }

    #[test]
    fn test_mqueue_try_wait() {
        let queue_name = "/test_bridge_queue3";

        let mq = BridgeSemaphore::create_with_name(queue_name).expect("Failed to create queue");

        // Try wait without signal - should return false
        let result = mq.try_wait().expect("Failed to try_wait");
        assert!(!result);

        // Post signal
        mq.post().expect("Failed to post");

        // Try wait with signal - should return true
        let result = mq.try_wait().expect("Failed to try_wait");
        assert!(result);

        // Try wait again - should return false (signal consumed)
        let result = mq.try_wait().expect("Failed to try_wait");
        assert!(!result);
    }

    #[test]
    fn test_mqueue_multiple_waiters() {
        let queue_name = "/test_bridge_queue4";

        let mq = BridgeSemaphore::create_with_name(queue_name).expect("Failed to create queue");

        // Post twice (for two waiters)
        mq.post().expect("Failed to post");
        mq.post().expect("Failed to post");

        // Two opens of the same queue
        let waiter1 = BridgeSemaphore::open_with_name(queue_name).expect("Failed to open queue");
        let waiter2 = BridgeSemaphore::open_with_name(queue_name).expect("Failed to open queue");

        // Both should be able to wait once
        waiter1.wait().expect("Failed to wait");
        waiter2.wait().expect("Failed to wait");
    }
}
