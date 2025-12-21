use crate::errors::BridgeError;
use memmap2::{Mmap, MmapOptions};
use std::fs::File;
use std::path::Path;

pub struct MmapReader {
    _file: File,
    mmap: Mmap,
}

impl MmapReader {
    pub fn new(path: impl AsRef<Path>) -> Result<Self, BridgeError> {
        let file = File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };

        Ok(Self { _file: file, mmap })
    }

    pub fn buffer(&self) -> &[u8] {
        &self.mmap
    }
}
