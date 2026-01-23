use crate::macros::impl_mmap_writer_base;
use crate::mmap_writer::MmapWriter;
use crate::paths;
use crate::types::{BoundingBox, TraceMetadata};
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
        frame_number: u64,
        timestamp_ns: u64,
        camera_id: u32,
        detections: &[BoundingBox],
    ) -> Result<()> {
        self.write_with_trace_context(frame_number, timestamp_ns, camera_id, detections, None)
    }

    pub fn write_with_trace_context(
        &mut self,
        frame_number: u64,
        timestamp_ns: u64,
        camera_id: u32,
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

        // Create trace context vectors if provided
        let (trace_id_vec, span_id_vec, trace_flags) = match trace_ctx {
            Some(ctx) => (
                Some(self.builder.create_vector(&ctx.trace_id)),
                Some(self.builder.create_vector(&ctx.span_id)),
                ctx.trace_flags,
            ),
            None => (None, None, 0),
        };

        let detection_result = schema::DetectionResult::create(
            &mut self.builder,
            &schema::DetectionResultArgs {
                frame_number,
                timestamp_ns,
                camera_id,
                detections: Some(detection_offset),
                trace_id: trace_id_vec,
                span_id: span_id_vec,
                trace_flags,
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
