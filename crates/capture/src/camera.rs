use crate::config::CameraConfig;
use crate::decoder::{FrameDecoder, MjpegDecoder, YuyvDecoder};
use crate::device::{CameraDevice, PixelFormat};
use crate::pacing::CapturePacing;
use crate::sink::FrameSink;
use crate::source::FrameSource;
use anyhow::Result;
use bridge::{SentryControl, SentryMode, capture_current_trace};
use common::span;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

pub struct Camera {
    camera_id: u32,
    device: CameraDevice,
    decoder: Box<dyn FrameDecoder>,
    sink: FrameSink,
    sentry_mode_fps: f64,
}

impl Camera {
    pub fn build(config: CameraConfig) -> Result<Self> {
        let camera_id = config.camera_id;
        let device = CameraDevice::open(&config)?;

        let decoder: Box<dyn FrameDecoder> = match device.pixel_format {
            PixelFormat::Yuyv => Box::new(YuyvDecoder::new()),
            PixelFormat::Mjpeg => Box::new(MjpegDecoder::new()?),
        };

        let sink = FrameSink::new()?;

        Ok(Self {
            camera_id,
            device,
            decoder,
            sink,
            sentry_mode_fps: config.sentry_mode_fps,
        })
    }

    pub fn run(&mut self, shutdown: &Arc<AtomicBool>, sentry: &SentryControl) -> Result<()> {
        tracing::info!(
            "Starting camera stream at {}x{}...",
            self.device.width,
            self.device.height,
        );

        let mut source = FrameSource::new(&self.device.device)?;
        let mut pacing = CapturePacing::new(self.device.max_fps, self.sentry_mode_fps);

        let mut frame_count = 0u64;
        let mut dropped_frames = 0u64;

        while !shutdown.load(Ordering::Relaxed) {
            let start_time = std::time::Instant::now();

            let mode = sentry.get_mode();
            if pacing.update(mode) {
                // Flush stale frames when entering Alarmed mode
                if mode == SentryMode::Alarmed {
                    let flushed = source.flush();
                    if flushed > 0 {
                        tracing::debug!("Flushed {} stale frames on mode transition", flushed);
                    }
                }
                tracing::info!(
                    "Sentry mode changed to {:?} ({:?})",
                    mode,
                    pacing.frame_duration()
                );
            }

            match source.next_frame() {
                Ok((buf, meta)) => {
                    let _s = span!("capture_frame");

                    // Decode directly using split borrow (decoder + sink are separate fields)
                    let rgb_data =
                        match self
                            .decoder
                            .decode(buf, self.device.width, self.device.height)
                        {
                            Ok(data) => data,
                            Err(e) => {
                                dropped_frames += 1;
                                tracing::warn!("Frame #{} decode error: {}", frame_count, e);
                                continue;
                            }
                        };

                    let trace_ctx = capture_current_trace();

                    if let Err(e) = self.sink.write(
                        rgb_data,
                        self.camera_id,
                        frame_count,
                        self.device.width,
                        self.device.height,
                        trace_ctx.as_ref(),
                    ) {
                        dropped_frames += 1;
                        tracing::warn!("Frame #{} write error: {}", frame_count, e);
                    } else {
                        frame_count += 1;
                    }

                    if frame_count > 0 && frame_count.is_multiple_of(30) {
                        tracing::debug!(
                            "Status: [Frames: {}] [Dropped: {}] [Seq: {}] [V4L seq: {}] [Mode: {:?}]",
                            frame_count,
                            dropped_frames,
                            self.sink.sequence(),
                            meta.sequence,
                            pacing.mode()
                        );
                    }
                }
                Err(e) => {
                    dropped_frames += 1;
                    tracing::warn!("Frame #{} capture error: {}", frame_count, e);
                }
            }

            let elapsed = start_time.elapsed();
            if elapsed < pacing.frame_duration() {
                std::thread::sleep(pacing.frame_duration() - elapsed);
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
