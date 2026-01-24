use crate::{macros::impl_mmap_writer_base, mmap_writer::MmapWriter, paths};
use anyhow::{Context, Result};
use common::span;
use schema::{Frame, FrameArgs, TraceContext};
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
    pub fn write_frame(
        &mut self,
        camera_id: u32,
        pixel_data: &[u8],
        frame_count: u64,
        width: u32,
        height: u32,
        trace_ctx: Option<&TraceContext>,
    ) -> Result<()> {
        let _s = span!("write_frame");

        let timestamp_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .context("Time went backwards")?
            .as_nanos() as u64;

        self.builder.reset();
        let pixels_vec = self.builder.create_vector(pixel_data);

        let frame_fb = Frame::create(
            &mut self.builder,
            &FrameArgs {
                camera_id,
                frame_number: frame_count,
                timestamp_ns,
                width,
                height,
                channels: 3,
                pixels: Some(pixels_vec),
                trace: trace_ctx,
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
