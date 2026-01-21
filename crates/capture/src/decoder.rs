use anyhow::{Context, Result};

/// Trait for decoding raw camera frames to RGB
pub trait FrameDecoder: Send + Sync {
    /// Decode raw frame data to RGB (3 bytes per pixel)
    fn decode(&self, raw: &[u8], width: u32, height: u32) -> Result<Vec<u8>>;
}

/// YUYV (YUV 4:2:2) decoder
///
/// YUYV packs 2 pixels in 4 bytes: [Y0, U, Y1, V]
/// Handles stride padding by only processing expected bytes per row.
pub struct YuyvDecoder;

impl FrameDecoder for YuyvDecoder {
    #[tracing::instrument(skip(self, raw))]
    fn decode(&self, raw: &[u8], width: u32, height: u32) -> Result<Vec<u8>> {
        let pixel_count = (width * height) as usize;
        let mut rgb = Vec::with_capacity(pixel_count * 3);

        let bytes_per_row = (width * 2) as usize; // YUYV = 2 bytes per pixel
        let stride = raw.len() / height as usize; // Actual stride (may include padding)

        for row in 0..height as usize {
            let row_start = row * stride;
            let row_data = &raw[row_start..row_start + bytes_per_row];

            for chunk in row_data.chunks_exact(4) {
                // YUYV: [Y0, U, Y1, V]
                let y0 = chunk[0] as f32;
                let u = chunk[1] as f32 - 128.0;
                let y1 = chunk[2] as f32;
                let v = chunk[3] as f32 - 128.0;

                // First pixel
                let r0 = (y0 + 1.402 * v).clamp(0.0, 255.0) as u8;
                let g0 = (y0 - 0.344 * u - 0.714 * v).clamp(0.0, 255.0) as u8;
                let b0 = (y0 + 1.772 * u).clamp(0.0, 255.0) as u8;
                rgb.extend_from_slice(&[r0, g0, b0]);

                // Second pixel
                let r1 = (y1 + 1.402 * v).clamp(0.0, 255.0) as u8;
                let g1 = (y1 - 0.344 * u - 0.714 * v).clamp(0.0, 255.0) as u8;
                let b1 = (y1 + 1.772 * u).clamp(0.0, 255.0) as u8;
                rgb.extend_from_slice(&[r1, g1, b1]);
            }
        }

        Ok(rgb)
    }
}

/// MJPEG decoder using turbojpeg (libjpeg-turbo)
pub struct MjpegDecoder;

impl FrameDecoder for MjpegDecoder {
    #[tracing::instrument(skip(self, raw))]
    fn decode(&self, raw: &[u8], _width: u32, _height: u32) -> Result<Vec<u8>> {
        let image: turbojpeg::Image<Vec<u8>> =
            turbojpeg::decompress(raw, turbojpeg::PixelFormat::RGB)
                .context("Failed to decode MJPEG frame")?;

        Ok(image.pixels)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_yuyv_decoder_basic() {
        let decoder = YuyvDecoder;
        // 2x1 image: 2 pixels = 4 bytes YUYV
        // Y=128 (gray), U=128, V=128 (neutral chroma)
        let yuyv = vec![128, 128, 128, 128];
        let rgb = decoder.decode(&yuyv, 2, 1).unwrap();
        assert_eq!(rgb.len(), 6); // 2 pixels * 3 bytes
    }

    #[test]
    fn test_mjpeg_decoder_invalid_data() {
        let decoder = MjpegDecoder;
        let invalid = vec![0, 1, 2, 3];
        assert!(decoder.decode(&invalid, 640, 480).is_err());
    }
}
