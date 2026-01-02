use crate::config::CameraConfig;
use crate::serialization::FrameSerializer;
use bridge::FrameSemaphore;
use nokhwa::Camera as NokhwaCamera;
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{CameraIndex, RequestedFormat, RequestedFormatType};
use std::time::Duration;
use v4l::{Device, capability::Flags, context};

/// Retry a function with exponential backoff
///
/// # Arguments
/// * `f` - The function to retry
/// * `max_retries` - Maximum number of retry attempts
/// * `base_delay_ms` - Initial delay in milliseconds (doubles each retry)
/// * `operation_name` - Human-readable name for logging
fn retry_with_backoff<F, T, E>(
    mut f: F,
    max_retries: u32,
    base_delay_ms: u64,
    operation_name: &str,
) -> Result<T, E>
where
    F: FnMut() -> Result<T, E>,
    E: std::fmt::Display,
{
    for attempt in 0..max_retries {
        match f() {
            Ok(result) => return Ok(result),
            Err(e) => {
                if attempt < max_retries - 1 {
                    let delay_ms = base_delay_ms * 2_u64.pow(attempt);
                    tracing::warn!(
                        "{} failed (attempt {}/{}): {}. Retrying in {}ms...",
                        operation_name,
                        attempt + 1,
                        max_retries,
                        e,
                        delay_ms
                    );
                    std::thread::sleep(Duration::from_millis(delay_ms));
                } else {
                    tracing::error!(
                        "{} failed after {} attempts: {}",
                        operation_name,
                        max_retries,
                        e
                    );
                    return Err(e);
                }
            }
        }
    }
    unreachable!()
}

fn find_usable_camera() -> Option<u32> {
    context::enum_devices()
        .into_iter()
        .find(|dev| is_usable_camera(dev.path()))
        .map(|dev| dev.index() as u32)
}

fn is_usable_camera(path: &std::path::Path) -> bool {
    Device::with_path(path)
        .and_then(|dev| dev.query_caps())
        .map(|caps| caps.capabilities.contains(Flags::VIDEO_CAPTURE))
        .unwrap_or(false)
}

fn open_camera(
    index: u32,
    format: RequestedFormat,
) -> Result<(NokhwaCamera, u32), Box<dyn std::error::Error>> {
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
    let best_idx = find_usable_camera().ok_or("No usable video devices found")?;

    let mut cam = NokhwaCamera::new(CameraIndex::Index(best_idx), format)?;
    cam.open_stream()?;

    Ok((cam, best_idx))
}

pub struct Camera {
    camera_id: u32,
    cam: NokhwaCamera,
    width: u32,
    height: u32,
    frame_duration: Duration,
    frame_serializer: FrameSerializer,
    inference_semaphore: FrameSemaphore,
    gateway_semaphore: FrameSemaphore,
}

impl Camera {
    pub fn build(config: CameraConfig) -> Result<Self, Box<dyn std::error::Error>> {
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
        let frame_duration = Duration::from_secs_f64(1.0 / format.frame_rate() as f64);

        let frame_serializer = FrameSerializer::build(&config.mmap_path, config.mmap_size)?;

        tracing::info!(
            "Created mmap at {} ({} MB)",
            config.mmap_path,
            config.mmap_size / 1024 / 1024
        );

        let get_sem = |path, name| {
            FrameSemaphore::open(path).or_else(|_| {
                tracing::info!("Creating new {} semaphore", name);
                FrameSemaphore::create(path)
            })
        };

        Ok(Self {
            camera_id: config.camera_id,
            cam,
            width: format.width(),
            height: format.height(),
            frame_duration,
            frame_serializer,
            inference_semaphore: get_sem("/bridge_frame_inference", "inference")?,
            gateway_semaphore: get_sem("/bridge_frame_gateway", "gateway")?,
        })
    }

    pub fn run(
        &mut self,
        shutdown: &std::sync::Arc<std::sync::atomic::AtomicBool>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        tracing::info!("frame buffer ready - writing at camera rate");

        let mut frame_count = 0u64;
        let mut dropped_frames = 0u64;

        while !shutdown.load(std::sync::atomic::Ordering::Relaxed) {
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
            if let Err(e) = self.gateway_semaphore.post() {
                tracing::warn!("Failed to signal gateway: {}", e);
            }

            frame_count += 1;

            if frame_count.is_multiple_of(30) {
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

        tracing::info!(
            "Shutdown signal received. Captured {} frames, dropped {}. Releasing camera...",
            frame_count,
            dropped_frames
        );

        Ok(())
    }
}
