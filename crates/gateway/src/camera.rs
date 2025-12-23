use bridge::FrameWriter;
use nokhwa::Camera as NokhwaCamera;
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{CameraIndex, RequestedFormat, RequestedFormatType};
use schema::{ColorFormat, FrameArgs};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

pub struct CameraConfig {
    pub device_id: u32,
    pub mmap_path: String,
    pub mmap_size: usize,
}

pub struct Camera {
    cam: NokhwaCamera,
    width: u32,
    height: u32,
    frame_duration: Duration,
    writer: FrameWriter,
    builder: flatbuffers::FlatBufferBuilder<'static>,
}

impl Camera {
    pub fn build(config: CameraConfig) -> Result<Self, Box<dyn std::error::Error>> {
        tracing::info!(
            "starting camera capture from /dev/video{}",
            config.device_id
        );

        let index = CameraIndex::Index(config.device_id);
        let requested_format =
            RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);

        let mut cam = NokhwaCamera::new(index, requested_format)?;
        cam.open_stream()?;

        tracing::info!("camera opened successfully");

        let camera_format = cam.camera_format();
        let width = camera_format.width();
        let height = camera_format.height();
        let fps = camera_format.frame_rate();
        let frame_duration = std::time::Duration::from_secs_f64(1.0 / fps as f64);

        tracing::info!(
            "camera properties: Resolution: {}x{}, FPS: {}",
            width,
            height,
            fps
        );

        let writer = FrameWriter::new(&config.mmap_path, config.mmap_size)?;
        tracing::info!(
            "Created mmap at {} ({} MB)",
            config.mmap_path,
            config.mmap_size / 1024 / 1024
        );

        let builder = flatbuffers::FlatBufferBuilder::new();

        Ok(Self {
            cam,
            width,
            height,
            frame_duration,
            writer,
            builder,
        })
    }

    pub fn run(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        tracing::info!("frame buffer ready - writing at camera rate");

        let mut frame_count = 0u64;

        loop {
            let frame = self.cam.frame()?;
            let decoded = frame.decode_image::<RgbFormat>()?;
            let pixel_data = decoded.as_raw();

            let timestamp_ns = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos() as u64;

            self.builder.reset();
            let pixels_vec = self.builder.create_vector(pixel_data);

            let frame_fb = schema::Frame::create(
                &mut self.builder,
                &FrameArgs {
                    frame_number: frame_count,
                    timestamp_ns,
                    camera_id: 0,
                    width: self.width,
                    height: self.height,
                    channels: 3,
                    format: ColorFormat::RGB,
                    pixels: Some(pixels_vec),
                },
            );

            self.builder.finish(frame_fb, None);
            let data = self.builder.finished_data();

            self.writer.write(data)?;

            frame_count += 1;

            if frame_count % 30 == 0 {
                tracing::debug!(
                    "Frame #{} (seq: {}), Size: {}x{}",
                    frame_count,
                    self.writer.sequence(),
                    self.width,
                    self.height
                );
            }

            std::thread::sleep(self.frame_duration);
        }
    }
}
