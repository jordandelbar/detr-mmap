use crate::{macros::impl_mmap_writer_base, mmap_writer::MmapWriter, paths};
use anyhow::{Context, Result};
use schema::{ColorFormat, FrameArgs};
use std::time::{SystemTime, UNIX_EPOCH};

pub struct FrameWriter {
    writer: MmapWriter,
    builder: flatbuffers::FlatBufferBuilder<'static>,
}

impl_mmap_writer_base!(FrameWriter, paths::FRAME_BUFFER_PATH, paths::DEFAULT_FRAME_BUFFER_SIZE);

impl FrameWriter {
    pub fn write(
        &mut self,
        pixel_data: &[u8],
        camera_id: u32,
        frame_count: u64,
        width: u32,
        height: u32,
    ) -> Result<()> {
        let timestamp_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .context("Time went backwards")?
            .as_nanos() as u64;

        self.builder.reset();
        let pixels_vec = self.builder.create_vector(pixel_data);

        let frame_fb = schema::Frame::create(
            &mut self.builder,
            &FrameArgs {
                frame_number: frame_count,
                timestamp_ns,
                camera_id,
                width,
                height,
                channels: 3,
                format: ColorFormat::RGB,
                pixels: Some(pixels_vec),
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
