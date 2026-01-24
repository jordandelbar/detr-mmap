use anyhow::Result;
use common::span;

/// Trait for decoding raw camera frames to RGB.
pub trait FrameDecoder: Send {
    /// Decode raw frame data to RGB (3 bytes per pixel).
    /// Returns a reference to the decoder's internal buffer.
    fn decode(&mut self, raw: &[u8], width: u32, height: u32) -> Result<&[u8]>;
}

/// YUYV (YUV 4:2:2) decoder.
///
/// YUYV packs 2 pixels in 4 bytes: [Y0, U, Y1, V]
pub struct YuyvDecoder {
    rgb_buffer: Vec<u8>,
}

impl Default for YuyvDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl YuyvDecoder {
    pub fn new() -> Self {
        Self {
            rgb_buffer: vec![0u8; 1920 * 1080 * 3],
        }
    }
}

impl FrameDecoder for YuyvDecoder {
    fn decode(&mut self, raw: &[u8], width: u32, height: u32) -> Result<&[u8]> {
        let _s = span!("decode");

        let pixel_count = (width * height) as usize;
        let rgb_size = pixel_count * 3;

        if self.rgb_buffer.len() < rgb_size {
            self.rgb_buffer.resize(rgb_size, 0);
        }

        let bytes_per_row = (width * 2) as usize;
        let stride = raw.len() / height as usize;

        let mut out_idx = 0;
        for row in 0..height as usize {
            let row_start = row * stride;
            let row_data = &raw[row_start..row_start + bytes_per_row];

            for chunk in row_data.chunks_exact(4) {
                // YUYV: [Y0, U, Y1, V]
                let y0 = chunk[0] as i32;
                let u = chunk[1] as i32 - 128;
                let y1 = chunk[2] as i32;
                let v = chunk[3] as i32 - 128;

                // BT.601 fixed-point coefficients (8-bit fraction)
                // R = Y + 1.402*V  -> Y + (359*V >> 8)
                // G = Y - 0.344*U - 0.714*V -> Y - ((88*U + 183*V) >> 8)
                // B = Y + 1.772*U -> Y + (454*U >> 8)
                let rv = (359 * v) >> 8;
                let gu = (88 * u + 183 * v) >> 8;
                let bu = (454 * u) >> 8;

                // First pixel
                self.rgb_buffer[out_idx] = (y0 + rv).clamp(0, 255) as u8;
                self.rgb_buffer[out_idx + 1] = (y0 - gu).clamp(0, 255) as u8;
                self.rgb_buffer[out_idx + 2] = (y0 + bu).clamp(0, 255) as u8;
                out_idx += 3;

                // Second pixel
                self.rgb_buffer[out_idx] = (y1 + rv).clamp(0, 255) as u8;
                self.rgb_buffer[out_idx + 1] = (y1 - gu).clamp(0, 255) as u8;
                self.rgb_buffer[out_idx + 2] = (y1 + bu).clamp(0, 255) as u8;
                out_idx += 3;
            }
        }

        Ok(&self.rgb_buffer[..rgb_size])
    }
}

/// MJPEG decoder using turbojpeg (libjpeg-turbo)
pub struct MjpegDecoder {
    decompressor: turbojpeg::Decompressor,
    rgb_buffer: Vec<u8>,
}

impl Default for MjpegDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl MjpegDecoder {
    pub fn new() -> Self {
        Self {
            decompressor: turbojpeg::Decompressor::new().unwrap(),
            rgb_buffer: vec![0u8; 1920 * 1080 * 3],
        }
    }
}

impl FrameDecoder for MjpegDecoder {
    fn decode(&mut self, raw: &[u8], _width: u32, _height: u32) -> Result<&[u8]> {
        let _s = span!("decode");

        let header = self.decompressor.read_header(raw)?;
        let width = header.width;
        let height = header.height;
        let rgb_size = width * height * 3;

        if self.rgb_buffer.len() < rgb_size {
            self.rgb_buffer.resize(rgb_size, 0);
        }

        let output = turbojpeg::Image {
            pixels: &mut self.rgb_buffer[..rgb_size],
            width,
            pitch: width * 3,
            height,
            format: turbojpeg::PixelFormat::RGB,
        };

        self.decompressor.decompress(raw, output)?;

        Ok(&self.rgb_buffer[..rgb_size])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_yuyv_decoder_basic() {
        let mut decoder = YuyvDecoder::new();
        // 2x1 image: 2 pixels = 4 bytes YUYV
        // Y=128 (gray), U=128, V=128 (neutral chroma)
        let yuyv = vec![128, 128, 128, 128];
        let rgb = decoder.decode(&yuyv, 2, 1).unwrap();
        assert_eq!(rgb.len(), 6); // 2 pixels * 3 bytes
    }

    #[test]
    fn test_mjpeg_decoder_invalid_data() {
        let mut decoder = MjpegDecoder::new();
        let invalid = vec![0, 1, 2, 3];
        assert!(decoder.decode(&invalid, 640, 480).is_err());
    }
}
