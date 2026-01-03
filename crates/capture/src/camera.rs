use crate::config::CameraConfig;
use crate::serialization::FrameSerializer;
use anyhow::{Context, Result, anyhow};
use bridge::{FrameSemaphore, SentryControl, SentryMode};
use common::retry::retry_with_backoff;
use nokhwa::Camera as NokhwaCamera;
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{CameraIndex, RequestedFormat, RequestedFormatType};
use std::{
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};
use v4l::{Device, capability::Flags, context};

fn find_usable_camera() -> Option<u32> {
    context::enum_devices()
        .into_iter()
        .find(|dev| is_usable_camera(dev.path()))
        .map(|dev| dev.index() as u32)
}

fn is_usable_camera(path: &Path) -> bool {
    Device::with_path(path)
        .and_then(|dev| dev.query_caps())
        .map(|caps| caps.capabilities.contains(Flags::VIDEO_CAPTURE))
        .unwrap_or(false)
}

fn open_camera(
    index: u32,
    format: RequestedFormat,
) -> Result<(NokhwaCamera, u32)> {
    let cam_index = CameraIndex::Index(index);
    if let Ok(mut cam) = NokhwaCamera::new(cam_index, format) {
        if cam.open_stream().is_ok() {
            return Ok((cam, index));
        }
    }

    tracing::debug!(
        "Camera index {} busy or missing, scanning alternatives...",
        index
    );
    let best_idx = find_usable_camera().ok_or_else(|| anyhow!("No usable video devices found"))?;

    let mut cam = NokhwaCamera::new(CameraIndex::Index(best_idx), format)?;
    cam.open_stream()?;

    Ok((cam, best_idx))
}

pub struct Camera {
    camera_id: u32,
    cam: NokhwaCamera,
    width: u32,
    height: u32,
    max_frame_rate: f64,
    sentry_mode_rate: f64,
    frame_serializer: FrameSerializer,
    inference_semaphore: FrameSemaphore,
    gateway_semaphore: FrameSemaphore,
}

impl Camera {
    pub fn build(config: CameraConfig) -> Result<Self> {
        let requested_format =
            RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);

        let (cam, actual_device_id) = retry_with_backoff(
            || open_camera(config.device_id, requested_format),
            10,
            200,
            "Camera Init",
        )?;

        tracing::info!(
            "Camera opened successfully at /dev/video{}",
            actual_device_id
        );

        let format = cam.camera_format();
        let frame_serializer = FrameSerializer::build(&config.mmap_path, config.mmap_size)
            .context("Failed to initialize frame serializer")?;

        tracing::info!(
            "Created mmap at {} ({} MB)",
            config.mmap_path,
            config.mmap_size / 1024 / 1024
        );

        let get_sem = |path: &str, name: &str| -> Result<FrameSemaphore> {
            FrameSemaphore::open(path).or_else(|_| {
                tracing::info!("Creating new {} semaphore", name);
                FrameSemaphore::create(path).map_err(|e| anyhow!("Failed to create semaphore {}: {}", name, e))
            })
        };

        Ok(Self {
            camera_id: config.camera_id,
            cam,
            width: format.width(),
            height: format.height(),
            max_frame_rate: format.frame_rate() as f64,
            sentry_mode_rate: config.sentry_mode_fps,
            frame_serializer,
            inference_semaphore: get_sem(&config.inference_semaphore_name, "inference")?,
            gateway_semaphore: get_sem(&config.gateway_semaphore_name, "gateway")?,
        })
    }

    fn process_single_frame(
        &mut self,
        frame_count: u64,
    ) -> Result<usize> {
        let frame = self.cam.frame()?;
        let decoded = frame.decode_image::<RgbFormat>()?;
        let pixel_data = decoded.as_raw();

        self.frame_serializer.write(
            pixel_data,
            self.camera_id,
            frame_count,
            self.width,
            self.height,
        )?;

        Ok(pixel_data.len())
    }

    pub fn run(
        &mut self,
        shutdown: &Arc<AtomicBool>,
        sentry: &SentryControl,
    ) -> Result<()> {
        tracing::info!(
            "Starting camera stream at {}x{}...",
            self.width,
            self.height
        );

        let mut frame_count = 0u64;
        let mut dropped_frames = 0u64;
        let mut current_mode = SentryMode::Standby;
        let standby_duration = Duration::from_secs_f64(1.0 / self.sentry_mode_rate);
        let alarmed_duration = Duration::from_secs_f64(1.0 / self.max_frame_rate);
        let mut frame_duration = standby_duration;

        while !shutdown.load(Ordering::Relaxed) {
            let start_time = std::time::Instant::now();

            let mode = sentry.get_mode();
            if mode != current_mode {
                current_mode = mode;
                frame_duration = match mode {
                    SentryMode::Standby => standby_duration,
                    SentryMode::Alarmed => alarmed_duration,
                };
                tracing::info!("Sentry mode changed to {:?} ({:?})", mode, frame_duration);
            }

            match self.process_single_frame(frame_count) {
                Ok(_bytes) => {
                    let _ = self.inference_semaphore.post();
                    let _ = self.gateway_semaphore.post();

                    frame_count += 1;
                }
                Err(e) => {
                    dropped_frames += 1;
                    tracing::warn!("Frame #{} pipeline error: {}", frame_count, e);
                }
            };

            if frame_count > 0 && frame_count % 30 == 0 {
                tracing::debug!(
                    "Status: [Frames: {}] [Dropped: {}] [Seq: {}] [Mode: {:?}]",
                    frame_count,
                    dropped_frames,
                    self.frame_serializer.sequence(),
                    current_mode
                );
            }

            let elapsed = start_time.elapsed();
            if elapsed < frame_duration {
                std::thread::sleep(frame_duration - elapsed);
            } else {
                tracing::trace!("Processing took longer than frame budget: {:?}", elapsed);
            }
        }

        tracing::info!(
            "Shutdown: {} frames captured, {} dropped.",
            frame_count,
            dropped_frames
        );
        Ok(())
    }
}
