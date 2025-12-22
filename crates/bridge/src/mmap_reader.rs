use crate::errors::BridgeError;
use memmap2::{Mmap, MmapOptions};
use std::fs::File;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

const DATA_OFFSET: usize = 8;

pub struct MmapReader {
    _file: File,
    mmap: Mmap,
    last_sequence: u64,
}

impl MmapReader {
    pub fn new(path: impl AsRef<Path>) -> Result<Self, BridgeError> {
        let file = File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };

        Ok(Self {
            _file: file,
            mmap,
            last_sequence: 0,
        })
    }

    /// Returns the current sequence number from mmap
    pub fn current_sequence(&self) -> u64 {
        let seq_ptr = self.mmap.as_ptr() as *const AtomicU64;
        unsafe { (*seq_ptr).load(Ordering::Acquire) }
    }

    /// Checks if new data is available (sequence number changed)
    pub fn has_new_data(&self) -> bool {
        let seq = self.current_sequence();
        seq > 0 && seq > self.last_sequence
    }

    /// Returns data buffer (skips the 8-byte sequence header)
    pub fn buffer(&self) -> &[u8] {
        &self.mmap[DATA_OFFSET..]
    }

    /// Mark current sequence as read
    pub fn mark_read(&mut self) {
        self.last_sequence = self.current_sequence();
    }

    /// Get last read sequence number
    pub fn last_sequence(&self) -> u64 {
        self.last_sequence
    }
}
