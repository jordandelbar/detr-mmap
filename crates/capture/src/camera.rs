use crate::config::CameraConfig;
use crate::serialization::FrameSerializer;
use bridge::FrameSemaphore;
use nokhwa::Camera as NokhwaCamera;
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{CameraIndex, RequestedFormat, RequestedFormatType};
use std::time::Duration;

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

/// Scan for available video devices
///
/// Tries device indices 0-9 and returns the first available camera.
/// This handles cases where cameras reconnect as /dev/video1, /dev/video2, etc.
fn find_available_camera(
    requested_format: &RequestedFormat,
) -> Result<(NokhwaCamera, u32), Box<dyn std::error::Error>> {
    // Try indices 0-9 (covers /dev/video0 through /dev/video9)
    for device_idx in 0..10 {
        let index = CameraIndex::Index(device_idx);

        match NokhwaCamera::new(index, *requested_format) {
            Ok(mut cam) => match cam.open_stream() {
                Ok(_) => {
                    tracing::info!("Found available camera at /dev/video{}", device_idx);
                    return Ok((cam, device_idx));
                }
                Err(e) => {
                    tracing::debug!(
                        "Camera at /dev/video{} exists but failed to open: {}",
                        device_idx,
                        e
                    );
                    continue;
                }
            },
            Err(_) => {
                continue;
            }
        }
    }

    Err("No available camera found in /dev/video0-9".into())
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

        // Wrap camera opening in retry logic to handle kernel delays after pod restart
        let (cam, actual_device_id) = retry_with_backoff(
            || {
                // Try configured device first
                tracing::info!(
                    "Attempting to open camera at /dev/video{}",
                    config.device_id
                );

                let index = CameraIndex::Index(config.device_id);
                match NokhwaCamera::new(index, requested_format) {
                    Ok(mut cam) => match cam.open_stream() {
                        Ok(_) => {
                            tracing::info!(
                                "Successfully opened configured camera /dev/video{}",
                                config.device_id
                            );
                            Ok((cam, config.device_id))
                        }
                        Err(e) => {
                            tracing::debug!(
                                "Configured camera /dev/video{} busy: {}. Scanning alternatives...",
                                config.device_id,
                                e
                            );
                            find_available_camera(&requested_format)
                        }
                    },
                    Err(e) => {
                        tracing::debug!(
                            "Configured camera /dev/video{} not found: {}. Scanning alternatives...",
                            config.device_id,
                            e
                        );
                        find_available_camera(&requested_format)
                    }
                }
            },
            10,  // max retries
            200, // base delay ms (200ms, 400ms, 800ms, 1600ms, 3200ms...)
            "Camera initialization",
        )?;

        tracing::info!(
            "Camera opened successfully at /dev/video{}",
            actual_device_id
        );

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

        // Try to open existing semaphores first (allows seamless restart)
        let inference_semaphore = match FrameSemaphore::open("/bridge_frame_inference") {
            Ok(sem) => {
                tracing::info!("Reconnected to existing inference semaphore");
                sem
            }
            Err(_) => {
                tracing::info!("Creating new inference semaphore");
                FrameSemaphore::create("/bridge_frame_inference")?
            }
        };

        let gateway_semaphore = match FrameSemaphore::open("/bridge_frame_gateway") {
            Ok(sem) => {
                tracing::info!("Reconnected to existing gateway semaphore");
                sem
            }
            Err(_) => {
                tracing::info!("Creating new gateway semaphore");
                FrameSemaphore::create("/bridge_frame_gateway")?
            }
        };

        Ok(Self {
            camera_id: config.camera_id,
            cam,
            width,
            height,
            frame_duration,
            frame_serializer,
            inference_semaphore,
            gateway_semaphore,
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
