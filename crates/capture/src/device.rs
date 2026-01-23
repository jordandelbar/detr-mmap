use crate::config::CameraConfig;
use anyhow::{Context, Result, anyhow};
use common::retry::retry_with_backoff;
use v4l::{
    Device, FourCC,
    control::{Control, Value},
    video::Capture,
};

const FOURCC_YUYV: FourCC = FourCC { repr: *b"YUYV" };
const FOURCC_MJPG: FourCC = FourCC { repr: *b"MJPG" };

// V4L2 control IDs (from videodev2.h)
const V4L2_CID_EXPOSURE_AUTO: u32 = 0x009a0901;
const V4L2_CID_EXPOSURE_ABSOLUTE: u32 = 0x009a0902;

// Exposure auto mode: aperture priority allows auto-exposure with an upper limit
const V4L2_EXPOSURE_APERTURE_PRIORITY: i64 = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
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

/// Configure camera for crisp motion capture (fast shutter, no temporal blending)
fn configure_for_crisp_motion(device: &Device) {
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

    // In aperture priority, exposure_absolute sets the upper limit
    if has_exposure_absolute
        && let Some(ctrl_desc) = controls.iter().find(|c| c.id == V4L2_CID_EXPOSURE_ABSOLUTE)
    {
        // Cap at ~20ms (200 units) - allows decent brightness while limiting blur
        let max_exposure = 200i64;
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

pub struct CameraDevice {
    pub device: Device,
    pub width: u32,
    pub height: u32,
    pub pixel_format: PixelFormat,
    pub max_fps: f64,
}

impl CameraDevice {
    pub fn open(config: &CameraConfig) -> Result<Self> {
        let device = retry_with_backoff(|| open_device(config.device_id), 10, 200, "Camera init")?;

        let caps = device.query_caps()?;
        tracing::info!("Camera opened: {} ({})", caps.card, caps.driver);

        let pixel_format = select_format(&device)?;
        let fourcc = match pixel_format {
            PixelFormat::Yuyv => FOURCC_YUYV,
            PixelFormat::Mjpeg => FOURCC_MJPG,
        };

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

        configure_for_crisp_motion(&device);

        let params = device.params()?;
        let fps = params.interval.denominator as f64 / params.interval.numerator as f64;
        tracing::info!("Frame rate: {:.1} fps", fps);

        Ok(Self {
            device,
            width: format.width,
            height: format.height,
            pixel_format,
            max_fps: fps,
        })
    }
}
