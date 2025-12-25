use crate::errors::BridgeError;
use nix::mqueue::{
    MQ_OFlag, MqAttr, MqdT, mq_close, mq_open, mq_receive, mq_send, mq_timedreceive, mq_unlink,
};
use nix::sys::stat::Mode;
use nix::sys::time::TimeSpec;
use std::ffi::CString;

/// A wrapper around POSIX message queues for frame synchronization
///
/// This uses message queues to signal when new frames are available.
/// The gateway posts to the queue after writing each frame,
/// signaling both inference and logic processes.
pub struct FrameSemaphore {
    mqd: Option<MqdT>,
    name: CString,
    is_owner: bool,
}

unsafe impl Send for FrameSemaphore {}
unsafe impl Sync for FrameSemaphore {}

impl FrameSemaphore {
    /// Create a new message queue
    ///
    /// This will create a new message queue or open an existing one.
    /// The queue can hold up to 10 messages (signals).
    ///
    /// # Arguments
    /// * `name` - Name of the message queue (e.g., "/bridge_frame_ready")
    ///
    /// # Returns
    /// A new FrameSemaphore instance that owns the message queue
    pub fn create(name: &str) -> Result<Self, BridgeError> {
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

        Ok(Self {
            mqd: Some(mqd),
            name: c_name,
            is_owner: true,
        })
    }

    /// Open an existing message queue (for consumers/inference/logic)
    ///
    /// # Arguments
    /// * `name` - Name of the message queue (e.g., "/bridge_frame_ready")
    ///
    /// # Returns
    /// A new FrameSemaphore instance connected to the existing queue
    pub fn open(name: &str) -> Result<Self, BridgeError> {
        let c_name = CString::new(name)
            .map_err(|e| BridgeError::SemaphoreError(format!("Invalid queue name: {}", e)))?;

        // Open existing message queue
        let mqd = mq_open(c_name.as_c_str(), MQ_OFlag::O_RDWR, Mode::empty(), None)
            .map_err(|e| BridgeError::SemaphoreError(format!("Failed to open queue: {}", e)))?;

        Ok(Self {
            mqd: Some(mqd),
            name: c_name,
            is_owner: false,
        })
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
    /// It should be called twice per frame (once for inference, once for logic)
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

impl Drop for FrameSemaphore {
    fn drop(&mut self) {
        // Close the message queue
        if let Some(mqd) = self.mqd.take() {
            let _ = mq_close(mqd);
        }

        // Only unlink if this instance created the queue
        if self.is_owner {
            let _ = mq_unlink(self.name.as_c_str());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mqueue_create_and_open() {
        let queue_name = "/test_bridge_queue1";

        // Create queue
        let creator = FrameSemaphore::create(queue_name).expect("Failed to create queue");

        // Open existing queue
        let opener = FrameSemaphore::open(queue_name).expect("Failed to open queue");

        // Post from creator
        creator.post().expect("Failed to post");

        // Wait from opener
        opener.wait().expect("Failed to wait");

        // Cleanup happens automatically via Drop
    }

    #[test]
    fn test_mqueue_drain() {
        let queue_name = "/test_bridge_queue2";

        let mq = FrameSemaphore::create(queue_name).expect("Failed to create queue");

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

        let mq = FrameSemaphore::create(queue_name).expect("Failed to create queue");

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

        let mq = FrameSemaphore::create(queue_name).expect("Failed to create queue");

        // Post twice (for two waiters)
        mq.post().expect("Failed to post");
        mq.post().expect("Failed to post");

        // Two opens of the same queue
        let waiter1 = FrameSemaphore::open(queue_name).expect("Failed to open queue");
        let waiter2 = FrameSemaphore::open(queue_name).expect("Failed to open queue");

        // Both should be able to wait once
        waiter1.wait().expect("Failed to wait");
        waiter2.wait().expect("Failed to wait");
    }
}
