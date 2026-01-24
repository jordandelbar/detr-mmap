use crate::macros::impl_mmap_writer_base;
use crate::mmap_writer::MmapWriter;
use crate::paths;
use crate::types::TraceMetadata;
use anyhow::{Context, Result};
use flatbuffers::{FlatBufferBuilder, ForwardsUOffset, Vector, WIPOffset};

pub struct DetectionWriter {
    writer: MmapWriter,
    builder: FlatBufferBuilder<'static>,
}

impl_mmap_writer_base!(
    DetectionWriter,
    paths::DETECTION_BUFFER_PATH,
    paths::DEFAULT_DETECTION_BUFFER_SIZE
);

impl DetectionWriter {
    /// Returns a mutable reference to the internal FlatBufferBuilder.
    /// Call `reset()` on the builder before building, and `commit()` after finishing.
    #[inline]
    pub fn builder(&mut self) -> &mut FlatBufferBuilder<'static> {
        &mut self.builder
    }

    /// Commit the finished FlatBuffer data to shared memory.
    /// Call this after `builder.finish(...)`.
    pub fn commit(&mut self) -> Result<()> {
        let data = self.builder.finished_data();
        self.writer
            .write(data)
            .context("Failed to write detection data")?;
        Ok(())
    }

    /// Build and write a DetectionResult with pre-built detection offsets.
    /// This is the zero-copy path where detections are built directly into the buffer.
    pub fn write(
        &mut self,
        camera_id: u32,
        frame_number: u64,
        timestamp_ns: u64,
        detections: WIPOffset<Vector<'_, ForwardsUOffset<schema::Detection<'_>>>>,
        trace_ctx: Option<&TraceMetadata>,
    ) -> Result<()> {
        let trace = trace_ctx
            .map(|ctx| schema::TraceContext::new(&ctx.trace_id, &ctx.span_id, ctx.trace_flags));

        let detection_result = schema::DetectionResult::create(
            &mut self.builder,
            &schema::DetectionResultArgs {
                camera_id,
                frame_number,
                timestamp_ns,
                detections: Some(detections),
                trace: trace.as_ref(),
            },
        );

        self.builder.finish(detection_result, None);
        self.commit()
    }
}
