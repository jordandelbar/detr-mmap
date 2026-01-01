use bridge::FrameWriter;
use schema::{ColorFormat, FrameArgs};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

pub struct FrameSerializer {
    writer: FrameWriter,
    builder: flatbuffers::FlatBufferBuilder<'static>,
}

impl FrameSerializer {
    pub fn build(mmap_path: &str, mmap_size: usize) -> Result<Self, Box<dyn std::error::Error>> {
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
        pixel_data: &[u8],
        camera_id: u32,
        frame_count: u64,
        width: u32,
        height: u32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let timestamp_ns = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos() as u64;

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

        self.writer.write(data)?;

        Ok(())
    }

    pub fn sequence(&self) -> u64 {
        self.writer.sequence()
    }
}
