use crate::errors::BridgeError;
use memmap2::{MmapMut, MmapOptions};
use std::fs::OpenOptions;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

const DATA_OFFSET: usize = 8;

pub struct FrameWriter {
    mmap: MmapMut,
    sequence: u64,
}

impl FrameWriter {
    pub fn new(path: impl AsRef<Path>, size: usize) -> Result<Self, BridgeError> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;

        file.set_len(size as u64)?;

        let mut mmap = unsafe { MmapOptions::new().map_mut(&file)? };

        // Initialize sequence number to 0
        let seq_ptr = mmap.as_mut_ptr() as *mut AtomicU64;
        unsafe {
            (*seq_ptr).store(0, Ordering::Release);
        }

        Ok(Self { mmap, sequence: 0 })
    }

    pub fn write(&mut self, data: &[u8]) -> Result<(), BridgeError> {
        let available_space = self.mmap.len() - DATA_OFFSET;
        if data.len() > available_space {
            return Err(BridgeError::SizeMismatch);
        }

        // Write data first
        self.mmap[DATA_OFFSET..DATA_OFFSET + data.len()].copy_from_slice(data);

        // Increment sequence and write atomically (signals data is ready)
        self.sequence += 1;
        let seq_ptr = self.mmap.as_mut_ptr() as *mut AtomicU64;
        unsafe {
            (*seq_ptr).store(self.sequence, Ordering::Release);
        }

        Ok(())
    }

    pub fn buffer_mut(&mut self) -> &mut [u8] {
        &mut self.mmap[DATA_OFFSET..]
    }

    pub fn flush(&mut self) -> Result<(), BridgeError> {
        self.mmap.flush()?;
        Ok(())
    }

    pub fn sequence(&self) -> u64 {
        self.sequence
    }
}
