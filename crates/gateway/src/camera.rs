use bridge::FrameWriter;
use opencv::{
    prelude::*,
    videoio::{self, CAP_ANY, VideoCapture},
};
use schema::{ColorFormat, FrameArgs};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

pub struct CameraConfig {
    pub device_id: u32,
    pub mmap_path: String,
    pub mmap_size: usize,
}

pub struct Camera {
    cam: VideoCapture,
    width: f64,
    height: f64,
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

        let cam = VideoCapture::new(config.device_id as i32, CAP_ANY)?;

        if !VideoCapture::is_opened(&cam)? {
            return Err(format!("Failed to open camera at /dev/video{}", config.device_id).into());
        }

        tracing::info!("camera opened successfully");

        let width = cam.get(videoio::CAP_PROP_FRAME_WIDTH)?;
        let height = cam.get(videoio::CAP_PROP_FRAME_HEIGHT)?;
        let fps = cam.get(videoio::CAP_PROP_FPS)?;
        let frame_duration = std::time::Duration::from_secs_f64(1.0 / fps);

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

        let mut frame = Mat::default();
        let mut frame_count = 0u64;

        loop {
            self.cam.read(&mut frame)?;

            if frame.empty() {
                tracing::warn!("empty frame received");
                continue;
            }

            let channels = frame.channels() as u8;
            let pixel_data = frame.data_bytes()?;
            let timestamp_ns = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos() as u64;

            self.builder.reset();
            let pixels_vec = self.builder.create_vector(pixel_data);

            let frame_fb = schema::Frame::create(
                &mut self.builder,
                &FrameArgs {
                    frame_number: frame_count,
                    timestamp_ns,
                    camera_id: 0,
                    width: self.width as u32,
                    height: self.height as u32,
                    channels,
                    format: ColorFormat::BGR,
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
