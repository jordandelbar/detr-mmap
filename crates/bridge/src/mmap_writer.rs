use memmap2::{MmapMut, MmapOptions};
use std::fs::OpenOptions;
use std::io;
use std::path::Path;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum BridgeError {
    #[error("IO error: {0}")]
    IoError(#[from] io::Error),

    #[error("FlatBuffer verification failed")]
    InvalidFlatBuffer,

    #[error("Buffer size mismatch")]
    SizeMismatch,
}

pub struct FrameWriter {
    mmap: MmapMut,
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

        let mmap = unsafe { MmapOptions::new().map_mut(&file)? };

        Ok(Self { mmap })
    }

    pub fn write(&mut self, data: &[u8]) -> Result<(), BridgeError> {
        if data.len() > self.mmap.len() {
            return Err(BridgeError::SizeMismatch);
        }

        self.mmap[..data.len()].copy_from_slice(data);
        Ok(())
    }

    pub fn buffer_mut(&mut self) -> &mut [u8] {
        &mut self.mmap
    }

    pub fn flush(&mut self) -> Result<(), BridgeError> {
        self.mmap.flush()?;
        Ok(())
    }
}
