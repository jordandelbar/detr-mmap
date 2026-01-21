use crate::config::CameraConfig;
use crate::decoder::{FrameDecoder, MjpegDecoder, YuyvDecoder};
use anyhow::{Context, Result, anyhow};
use bridge::{
    BridgeSemaphore, FrameWriter, SemaphoreType, SentryControl, SentryMode, TraceContext,
    TraceContextBytes,
};
use common::retry::retry_with_backoff;
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};
use v4l::{
    FourCC,
    buffer::Type,
    control::{Control, Value},
    io::{mmap::Stream, traits::CaptureStream},
    prelude::*,
    video::Capture,
};

const BUFFER_COUNT: u32 = 4;

// Common FourCC codes
const FOURCC_YUYV: FourCC = FourCC { repr: *b"YUYV" };
const FOURCC_MJPG: FourCC = FourCC { repr: *b"MJPG" };

// V4L2 control IDs (from videodev2.h)
const V4L2_CID_EXPOSURE_AUTO: u32 = 0x009a0901;
const V4L2_CID_EXPOSURE_ABSOLUTE: u32 = 0x009a0902;

// Exposure auto mode: aperture priority allows auto-exposure with an upper limit
const V4L2_EXPOSURE_APERTURE_PRIORITY: i64 = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PixelFormat {
    Yuyv,
    Mjpeg,
}

fn find_usable_camera() -> Option<u32> {
    v4l::context::enum_devices()
        .into_iter()
        .find(|dev| {
            Device::with_path(dev.path())
                .and_then(|d| d.query_caps())
                .map(|caps| {
                    caps.capabilities
                        .contains(v4l::capability::Flags::VIDEO_CAPTURE)
                })
                .unwrap_or(false)
        })
        .map(|dev| dev.index() as u32)
}

fn open_device(index: u32) -> Result<Device> {
    if let Ok(dev) = Device::new(index as usize)
        && dev.query_caps().is_ok()
    {
        return Ok(dev);
    }

    tracing::debug!(
        "Camera index {} busy or missing, scanning alternatives...",
        index
    );

    let best_idx = find_usable_camera().ok_or_else(|| anyhow!("No usable video devices found"))?;
    Device::new(best_idx as usize).context("Failed to open fallback camera device")
}

/// Configure camera for crisp motion capture (fast shutter, no temporal blending)
fn configure_for_crisp_motion(device: &Device) {
    // Query available controls
    let controls = match device.query_controls() {
        Ok(c) => c,
        Err(e) => {
            tracing::warn!("Failed to query camera controls: {}", e);
            return;
        }
    };

    let has_exposure_auto = controls.iter().any(|c| c.id == V4L2_CID_EXPOSURE_AUTO);
    let has_exposure_absolute = controls.iter().any(|c| c.id == V4L2_CID_EXPOSURE_ABSOLUTE);

    tracing::debug!(
        "Camera controls: exposure_auto={}, exposure_absolute={}",
        has_exposure_auto,
        has_exposure_absolute
    );

    // Try aperture priority mode - auto-exposure that respects our exposure limit
    // This is less aggressive than full manual and adapts to lighting
    if has_exposure_auto {
        if let Err(e) = device.set_control(Control {
            id: V4L2_CID_EXPOSURE_AUTO,
            value: Value::Integer(V4L2_EXPOSURE_APERTURE_PRIORITY),
        }) {
            tracing::debug!("Aperture priority mode not supported: {}", e);
        } else {
            tracing::info!("Exposure mode: aperture priority (auto with limits)");
        }
    }

    // Note: In aperture priority, exposure_absolute sets the upper limit
    // The camera will use shorter exposures when there's enough light
    if has_exposure_absolute
        && let Some(ctrl_desc) = controls.iter().find(|c| c.id == V4L2_CID_EXPOSURE_ABSOLUTE)
    {
        // Cap at ~20ms (200 units) - allows decent brightness while limiting blur
        // At 30fps this is about 60% of frame time
        let max_exposure = 200i64; // 20ms
        let exposure = max_exposure.min(ctrl_desc.maximum);

        if let Err(e) = device.set_control(Control {
            id: V4L2_CID_EXPOSURE_ABSOLUTE,
            value: Value::Integer(exposure),
        }) {
            tracing::debug!("Failed to set exposure limit: {}", e);
        } else {
            tracing::info!(
                "Exposure limit: {} ({}ms max)",
                exposure,
                exposure as f64 / 10.0
            );
        }
    }

    if !has_exposure_auto && !has_exposure_absolute {
        tracing::info!("Camera does not expose exposure controls");
    }
}

/// Select best pixel format: prefer YUYV (faster decode), fallback to MJPEG
fn select_format(device: &Device) -> Result<PixelFormat> {
    let formats = device.enum_formats()?;

    tracing::debug!("Available formats:");
    for fmt in &formats {
        tracing::debug!("  {:?}: {}", fmt.fourcc, fmt.description);
    }

    // Prefer YUYV for faster decoding
    if formats.iter().any(|f| f.fourcc == FOURCC_YUYV) {
        return Ok(PixelFormat::Yuyv);
    }

    // Fallback to MJPEG
    if formats.iter().any(|f| f.fourcc == FOURCC_MJPG) {
        return Ok(PixelFormat::Mjpeg);
    }

    Err(anyhow!(
        "Camera supports neither YUYV nor MJPEG - available: {:?}",
        formats.iter().map(|f| f.fourcc).collect::<Vec<_>>()
    ))
}

/// Number of frames to discard on mode transition to flush stale buffers
const FLUSH_FRAME_COUNT: usize = 4;

pub struct Camera {
    camera_id: u32,
    device: Device,
    width: u32,
    height: u32,
    decoder: Box<dyn FrameDecoder>,
    max_frame_rate: f64,
    sentry_mode_rate: f64,
    frame_writer: FrameWriter,
    inference_semaphore: BridgeSemaphore,
    gateway_semaphore: BridgeSemaphore,
}

impl Camera {
    pub fn build(config: CameraConfig) -> Result<Self> {
        let device = retry_with_backoff(|| open_device(config.device_id), 10, 200, "Camera Init")?;

        let caps = device.query_caps()?;
        tracing::info!("Camera opened: {} ({})", caps.card, caps.driver);

        // Select best available format
        let pixel_format = select_format(&device)?;
        let fourcc = match pixel_format {
            PixelFormat::Yuyv => FOURCC_YUYV,
            PixelFormat::Mjpeg => FOURCC_MJPG,
        };

        // Set the selected format
        let mut format = device.format()?;
        format.fourcc = fourcc;
        let format = device.set_format(&format)?;

        tracing::info!(
            "Capture format: {}x{} {:?} ({:?})",
            format.width,
            format.height,
            format.fourcc,
            pixel_format
        );

        // Create decoder for the selected format
        let decoder: Box<dyn FrameDecoder> = match pixel_format {
            PixelFormat::Yuyv => Box::new(YuyvDecoder),
            PixelFormat::Mjpeg => Box::new(MjpegDecoder),
        };

        // Configure for crisp motion (fast shutter, no motion blur)
        configure_for_crisp_motion(&device);

        // Get frame rate from device parameters
        let params = device.params()?;
        let frame_rate = params.interval.denominator as f64 / params.interval.numerator as f64;
        tracing::info!("Frame rate: {:.1} fps", frame_rate);

        let frame_writer = FrameWriter::build().context("Failed to initialize frame writer")?;

        Ok(Self {
            camera_id: config.camera_id,
            device,
            width: format.width,
            height: format.height,
            decoder,
            max_frame_rate: frame_rate,
            sentry_mode_rate: config.sentry_mode_fps,
            frame_writer,
            inference_semaphore: BridgeSemaphore::ensure(SemaphoreType::FrameCaptureToInference)?,
            gateway_semaphore: BridgeSemaphore::ensure(SemaphoreType::FrameCaptureToGateway)?,
        })
    }

    /// Discard buffered frames to ensure fresh captures after mode transition.
    fn flush_stale_frames(stream: &mut Stream) -> usize {
        let mut flushed = 0;
        for _ in 0..FLUSH_FRAME_COUNT {
            if stream.next().is_ok() {
                flushed += 1;
            } else {
                break;
            }
        }
        flushed
    }

    /// Decode raw frame buffer to RGB
    #[tracing::instrument(skip(self, raw))]
    fn decode_frame(&self, raw: &[u8]) -> Result<Vec<u8>> {
        self.decoder.decode(raw, self.width, self.height)
    }

    pub fn run(&mut self, shutdown: &Arc<AtomicBool>, sentry: &SentryControl) -> Result<()> {
        tracing::info!(
            "Starting camera stream at {}x{}...",
            self.width,
            self.height,
        );

        let mut stream = Stream::with_buffers(&self.device, Type::VideoCapture, BUFFER_COUNT)
            .context("Failed to create capture stream")?;

        let mut frame_count = 0u64;
        let mut dropped_frames = 0u64;
        let mut semaphore_failures = 0u64;
        let mut current_mode = SentryMode::Standby;
        let standby_duration = Duration::from_secs_f64(1.0 / self.sentry_mode_rate);
        let alarmed_duration = Duration::from_secs_f64(1.0 / self.max_frame_rate);
        let mut frame_duration = standby_duration;

        while !shutdown.load(Ordering::Relaxed) {
            let start_time = std::time::Instant::now();

            let mode = sentry.get_mode();
            if mode != current_mode {
                let old_mode = current_mode;
                current_mode = mode;
                frame_duration = match mode {
                    SentryMode::Standby => standby_duration,
                    SentryMode::Alarmed => alarmed_duration,
                };

                // Flush stale frames when entering Alarmed mode
                if old_mode == SentryMode::Standby && mode == SentryMode::Alarmed {
                    let flushed = Self::flush_stale_frames(&mut stream);
                    if flushed > 0 {
                        tracing::debug!("Flushed {} stale frames on mode transition", flushed);
                    }
                }

                tracing::info!("Sentry mode changed to {:?} ({:?})", mode, frame_duration);
            }

            match stream.next() {
                Ok((buf, meta)) => {
                    // Create a span for this frame capture
                    let span = tracing::info_span!(
                        "capture_frame",
                        frame_number = frame_count,
                        camera_id = self.camera_id
                    );
                    let _enter = span.enter();

                    // Decode to RGB
                    let rgb_data = match self.decode_frame(buf) {
                        Ok(data) => data,
                        Err(e) => {
                            dropped_frames += 1;
                            tracing::warn!("Frame #{} decode error: {}", frame_count, e);
                            continue;
                        }
                    };

                    // Capture trace context from the current span for propagation
                    let trace_ctx =
                        TraceContext::from_current().map(|ctx| TraceContextBytes::from(&ctx));

                    if let Err(e) = self.frame_writer.write_with_trace_context(
                        &rgb_data,
                        self.camera_id,
                        frame_count,
                        self.width,
                        self.height,
                        trace_ctx.as_ref(),
                    ) {
                        dropped_frames += 1;
                        tracing::warn!("Frame #{} write error: {}", frame_count, e);
                    } else {
                        if self.inference_semaphore.post().is_err() {
                            semaphore_failures += 1;
                        }
                        if self.gateway_semaphore.post().is_err() {
                            semaphore_failures += 1;
                        }
                        frame_count += 1;
                    }

                    if frame_count > 0 && frame_count.is_multiple_of(30) {
                        if semaphore_failures > 0 {
                            tracing::warn!(
                                "Status: [Frames: {}] [Dropped: {}] [Semaphore failures: {}] [Mode: {:?}]",
                                frame_count,
                                dropped_frames,
                                semaphore_failures,
                                current_mode
                            );
                        } else {
                            tracing::debug!(
                                "Status: [Frames: {}] [Dropped: {}] [Seq: {}] [V4L seq: {}] [Mode: {:?}]",
                                frame_count,
                                dropped_frames,
                                self.frame_writer.sequence(),
                                meta.sequence,
                                current_mode
                            );
                        }
                    }
                }
                Err(e) => {
                    dropped_frames += 1;
                    tracing::warn!("Frame #{} capture error: {}", frame_count, e);
                }
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
