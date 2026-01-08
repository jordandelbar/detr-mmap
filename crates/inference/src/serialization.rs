use bridge::{FrameWriter, types::Detection};
use std::path::Path;

pub struct DetectionSerializer {
    writer: FrameWriter,
    builder: flatbuffers::FlatBufferBuilder<'static>,
}

impl DetectionSerializer {
    pub fn build(mmap_path: &str, mmap_size: usize) -> anyhow::Result<Self> {
        let writer = if Path::new(mmap_path).exists() {
            FrameWriter::open_existing(mmap_path)?
        } else {
            FrameWriter::create_and_init(mmap_path, mmap_size)?
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
    ) -> anyhow::Result<()> {
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

        let detections_offset = self.builder.create_vector(&bbox_vec);

        let detection_result = schema::DetectionResult::create(
            &mut self.builder,
            &schema::DetectionResultArgs {
                frame_number,
                timestamp_ns,
                camera_id,
                detections: Some(detections_offset),
            },
        );

        self.builder.finish(detection_result, None);
        let data = self.builder.finished_data();

        self.writer.write(data)?;

        Ok(())
    }

    pub fn sequence(&self) -> u64 {
        self.writer.sequence()
    }
}
