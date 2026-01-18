/// Generates common MmapWriter boilerplate methods: `build()`, `build_with_path()`, `sequence()`
#[cfg(any(feature = "frame-writer", feature = "detection-writer"))]
macro_rules! impl_mmap_writer_base {
    ($struct_name:ident, $default_path:expr, $default_size:expr) => {
        impl $struct_name {
            pub fn build() -> anyhow::Result<Self> {
                Self::build_with_path($default_path, $default_size)
            }

            pub fn build_with_path(mmap_path: &str, mmap_size: usize) -> anyhow::Result<Self> {
                use anyhow::Context;
                use std::path::Path;

                let writer = if Path::new(mmap_path).exists() {
                    crate::mmap_writer::MmapWriter::open_existing(mmap_path)
                        .context("Failed to open existing mmap writer")?
                } else {
                    crate::mmap_writer::MmapWriter::create_and_init(mmap_path, mmap_size)
                        .context("Failed to create new mmap writer")?
                };
                let builder = flatbuffers::FlatBufferBuilder::new();
                Ok(Self { writer, builder })
            }

            pub fn sequence(&self) -> u64 {
                self.writer.sequence()
            }
        }
    };
}

/// Generates common MmapReader boilerplate methods: `build()`, `with_path()`, `current_sequence()`, `mark_read()`
#[cfg(any(feature = "frame-reader", feature = "detection-reader"))]
macro_rules! impl_mmap_reader_base {
    ($struct_name:ident, $default_path:expr) => {
        impl $struct_name {
            pub fn build() -> anyhow::Result<Self> {
                Self::with_path($default_path)
            }

            pub fn with_path(mmap_path: &str) -> anyhow::Result<Self> {
                let reader = crate::mmap_reader::MmapReader::build(mmap_path)?;
                Ok(Self { reader })
            }

            pub fn current_sequence(&self) -> u64 {
                self.reader.current_sequence()
            }

            pub fn mark_read(&mut self) {
                self.reader.mark_read();
            }
        }
    };
}

#[cfg(any(feature = "frame-reader", feature = "detection-reader"))]
pub(crate) use impl_mmap_reader_base;
#[cfg(any(feature = "frame-writer", feature = "detection-writer"))]
pub(crate) use impl_mmap_writer_base;
