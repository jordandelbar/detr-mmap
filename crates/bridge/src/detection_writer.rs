use crate::mmap_writer::MmapWriter;
use crate::paths;
use crate::types::Detection;
use anyhow::{Context, Result};
use std::path::Path;

pub struct DetectionWriter {
    writer: MmapWriter,
    builder: flatbuffers::FlatBufferBuilder<'static>,
}

impl DetectionWriter {
    /// Create a new DetectionWriter using the default detection buffer path and size
    pub fn build() -> Result<Self> {
        Self::build_with_path(
            paths::DETECTION_BUFFER_PATH,
            paths::DEFAULT_DETECTION_BUFFER_SIZE,
        )
    }

    /// Create a new DetectionWriter with custom path and size (useful for tests and benchmarks)
    pub fn build_with_path(mmap_path: &str, mmap_size: usize) -> Result<Self> {
        let writer = if Path::new(mmap_path).exists() {
            MmapWriter::open_existing(mmap_path).context("Failed to open existing mmap writer")?
        } else {
            MmapWriter::create_and_init(mmap_path, mmap_size)
                .context("Failed to create new mmap writer")?
        };

        let builder = flatbuffers::FlatBufferBuilder::new();
        Ok(Self { writer, builder })
    }

    pub fn write(
        &mut self,
        frame_number: u64,
        timestamp_ns: u64,
        camera_id: u32,
        detections: &[Detection],
    ) -> Result<()> {
        self.builder.reset();

        let bbox_vec: Vec<_> = detections
            .iter()
            .map(|d| {
                schema::BoundingBox::create(
                    &mut self.builder,
                    &schema::BoundingBoxArgs {
                        x1: d.x1,
                        y1: d.y1,
                        x2: d.x2,
                        y2: d.y2,
                        confidence: d.confidence,
                        class_id: d.class_id,
                    },
                )
            })
            .collect();

        let detection_offset = self.builder.create_vector(&bbox_vec);

        let detection_result = schema::DetectionResult::create(
            &mut self.builder,
            &schema::DetectionResultArgs {
                frame_number,
                timestamp_ns,
                camera_id,
                detections: Some(detection_offset),
            },
        );

        self.builder.finish(detection_result, None);
        let data = self.builder.finished_data();

        self.writer
            .write(data)
            .context("Failed to write detection data")?;

        Ok(())
    }

    pub fn sequence(&self) -> u64 {
        self.writer.sequence()
    }
}
