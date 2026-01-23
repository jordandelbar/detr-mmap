use crate::{macros::impl_mmap_writer_base, mmap_writer::MmapWriter, paths, types::TraceMetadata};
use anyhow::{Context, Result};
use common::span;
use schema::{ColorFormat, FrameArgs};
use std::time::{SystemTime, UNIX_EPOCH};

pub struct FrameWriter {
    writer: MmapWriter,
    builder: flatbuffers::FlatBufferBuilder<'static>,
}

impl_mmap_writer_base!(
    FrameWriter,
    paths::FRAME_BUFFER_PATH,
    paths::DEFAULT_FRAME_BUFFER_SIZE
);

impl FrameWriter {
    pub fn write(
        &mut self,
        camera_id: u32,
        pixel_data: &[u8],
        frame_count: u64,
        width: u32,
        height: u32,
        trace_ctx: Option<&TraceMetadata>,
    ) -> Result<()> {
        let _s = span!("write_with_trace_context");

        let timestamp_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .context("Time went backwards")?
            .as_nanos() as u64;

        self.builder.reset();
        let pixels_vec = self.builder.create_vector(pixel_data);

        // Create trace metadata if provided
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

        let frame_fb = schema::Frame::create(
            &mut self.builder,
            &FrameArgs {
                camera_id,
                frame_number: frame_count,
                timestamp_ns,
                width,
                height,
                channels: 3,
                format: ColorFormat::RGB,
                pixels: Some(pixels_vec),
                trace: trace_offset,
            },
        );

        self.builder.finish(frame_fb, None);
        let data = self.builder.finished_data();

        self.writer
            .write(data)
            .context("Failed to write frame data")?;

        Ok(())
    }
}
