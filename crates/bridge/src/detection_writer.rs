use crate::macros::impl_mmap_writer_base;
use crate::mmap_writer::MmapWriter;
use crate::paths;
use anyhow::{Context, Result};

pub struct DetectionWriter {
    writer: MmapWriter,
    builder: flatbuffers::FlatBufferBuilder<'static>,
}

impl_mmap_writer_base!(
    DetectionWriter,
    paths::DETECTION_BUFFER_PATH,
    paths::DEFAULT_DETECTION_BUFFER_SIZE
);

impl DetectionWriter {
    pub fn write(
        &mut self,
        camera_id: u32,
        frame_number: u64,
        timestamp_ns: u64,
        detections: &[BoundingBox],
        trace_ctx: Option<&TraceMetadata>,
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

        let trace_offset = trace_ctx.map(|ctx| {
            let trace_id_vec = self.builder.create_vector(&ctx.trace_id);
            let span_id_vec = self.builder.create_vector(&ctx.span_id);
            schema::TraceMetadata::create(
                &mut self.builder,
                &schema::TraceMetadataArgs {
                    trace_id: Some(trace_id_vec),
                    span_id: Some(span_id_vec),
                    trace_flags: ctx.trace_flags,
                },
            )
        });

        let detection_result = schema::DetectionResult::create(
            &mut self.builder,
            &schema::DetectionResultArgs {
                camera_id,
                frame_number,
                timestamp_ns,
                detections: Some(detection_offset),
                trace: trace_offset,
            },
        );

        self.builder.finish(detection_result, None);
        let data = self.builder.finished_data();

        self.writer
            .write(data)
            .context("Failed to write detection data")?;

        Ok(())
    }
}
