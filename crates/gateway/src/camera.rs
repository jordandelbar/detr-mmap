use crate::config::CameraConfig;
use crate::serialization::FrameSerializer;
use bridge::FrameSemaphore;
use nokhwa::Camera as NokhwaCamera;
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{CameraIndex, RequestedFormat, RequestedFormatType};
use std::time::Duration;

pub struct Camera {
    camera_id: u32,
    cam: NokhwaCamera,
    width: u32,
    height: u32,
    frame_duration: Duration,
    frame_serializer: FrameSerializer,
    inference_semaphore: FrameSemaphore,
    logic_semaphore: FrameSemaphore,
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

        let frame_serializer = FrameSerializer::build(&config.mmap_path, config.mmap_size)?;

        tracing::info!(
            "Created mmap at {} ({} MB)",
            config.mmap_path,
            config.mmap_size / 1024 / 1024
        );

        let inference_semaphore = FrameSemaphore::create("/bridge_frame_inference")?;
        let logic_semaphore = FrameSemaphore::create("/bridge_frame_logic")?;
        tracing::info!("Created frame synchronization semaphores (inference + logic)");

        Ok(Self {
            camera_id: config.camera_id,
            cam,
            width,
            height,
            frame_duration,
            frame_serializer,
            inference_semaphore,
            logic_semaphore,
        })
    }

    pub fn run(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        tracing::info!("frame buffer ready - writing at camera rate");

        let mut frame_count = 0u64;
        let mut dropped_frames = 0u64;

        loop {
            let frame = match self.cam.frame() {
                Ok(f) => f,
                Err(e) => {
                    tracing::warn!("Failed to capture frame: {}", e);
                    dropped_frames += 1;
                    std::thread::sleep(self.frame_duration);
                    continue;
                }
            };

            let decoded = match frame.decode_image::<RgbFormat>() {
                Ok(d) => d,
                Err(e) => {
                    tracing::warn!("Failed to decode frame: {}", e);
                    dropped_frames += 1;
                    std::thread::sleep(self.frame_duration);
                    continue;
                }
            };

            let pixel_data = decoded.as_raw();

            if let Err(e) = self.frame_serializer.write(
                pixel_data,
                self.camera_id,
                frame_count,
                self.width,
                self.height,
            ) {
                tracing::error!(
                    "Failed to write frame #{} (size: {} bytes): {}",
                    frame_count,
                    pixel_data.len(),
                    e
                );
                dropped_frames += 1;
                std::thread::sleep(self.frame_duration);
                continue;
            }

            // Signal each consumer's dedicated queue
            if let Err(e) = self.inference_semaphore.post() {
                tracing::warn!("Failed to signal inference: {}", e);
            }
            if let Err(e) = self.logic_semaphore.post() {
                tracing::warn!("Failed to signal logic: {}", e);
            }

            frame_count += 1;

            if frame_count % 30 == 0 {
                tracing::debug!(
                    "Frame #{} (seq: {}), Size: {}x{}, Dropped: {}",
                    frame_count,
                    self.frame_serializer.sequence(),
                    self.width,
                    self.height,
                    dropped_frames
                );
            }

            std::thread::sleep(self.frame_duration);
        }
    }
}
