use crate::config::DEFAULT_INPUT_SIZE;
use fast_image_resize::{FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer, images::Image};
use ndarray::{Array, IxDyn};
use std::default::Default;

const LETTERBOX_COLOR: u8 = 114;

/// ImageNet normalization constants (used by RF-DETR)
const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

pub struct PreProcessor {
    pub input_size: (u32, u32),
    rgb_buffer: Vec<u8>,
    letterboxed_buffer: Vec<u8>,
}

impl PreProcessor {
    pub fn new(input_size: (u32, u32)) -> Self {
        Self {
            input_size,
            rgb_buffer: Vec::with_capacity(1920 * 1080 * 3),
            letterboxed_buffer: vec![LETTERBOX_COLOR; (input_size.0 * input_size.1 * 3) as usize],
        }
    }

    pub fn preprocess_frame(
        &mut self,
        pixels: flatbuffers::Vector<u8>,
        width: u32,
        height: u32,
        format: bridge::ColorFormat,
    ) -> anyhow::Result<(Array<f32, IxDyn>, f32, f32, f32)> {
        tracing::trace!(
            width,
            height,
            format = ?format,
            pixel_bytes = pixels.len(),
            "Preprocessing frame dimensions"
        );

        let expected_size = (width * height * 3) as usize;

        if self.rgb_buffer.capacity() < expected_size {
            self.rgb_buffer
                .reserve(expected_size - self.rgb_buffer.len());
        }
        self.rgb_buffer.clear();

        match format {
            bridge::ColorFormat::RGB => {
                self.rgb_buffer.extend_from_slice(pixels.bytes());
            }
            bridge::ColorFormat::BGR => {
                for i in (0..pixels.len()).step_by(3) {
                    let b = pixels.get(i);
                    let g = pixels.get(i + 1);
                    let r = pixels.get(i + 2);
                    self.rgb_buffer.push(r);
                    self.rgb_buffer.push(g);
                    self.rgb_buffer.push(b);
                }
            }
            bridge::ColorFormat::GRAY => {
                return Err(anyhow::anyhow!("Grayscale format not supported"));
            }
            _ => {
                return Err(anyhow::anyhow!("Unknown color format"));
            }
        }

        if self.rgb_buffer.len() != expected_size {
            return Err(anyhow::anyhow!(
                "Buffer size mismatch: expected {} bytes for {}x{} RGB, got {} bytes",
                expected_size,
                width,
                height,
                self.rgb_buffer.len()
            ));
        }

        let scale =
            (self.input_size.0 as f32 / width as f32).min(self.input_size.1 as f32 / height as f32);
        let new_width = (width as f32 * scale) as u32;
        let new_height = (height as f32 * scale) as u32;
        let offset_x = (self.input_size.0 - new_width) / 2;
        let offset_y = (self.input_size.1 - new_height) / 2;

        let src_image = Image::from_slice_u8(width, height, &mut self.rgb_buffer, PixelType::U8x3)?;

        let mut dst_image = Image::new(new_width, new_height, PixelType::U8x3);

        let mut resizer = Resizer::new();
        resizer.resize(
            &src_image,
            &mut dst_image,
            &ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Bilinear)),
        )?;

        self.letterboxed_buffer.fill(LETTERBOX_COLOR);

        let resized_data = dst_image.buffer();
        for y in 0..new_height {
            let src_row_start = (y * new_width * 3) as usize;
            let src_row_end = src_row_start + (new_width * 3) as usize;
            let dst_row_start = ((y + offset_y) * self.input_size.0 * 3 + offset_x * 3) as usize;
            let dst_row_end = dst_row_start + (new_width * 3) as usize;

            self.letterboxed_buffer[dst_row_start..dst_row_end]
                .copy_from_slice(&resized_data[src_row_start..src_row_end]);
        }

        let mut input = Array::zeros(IxDyn(&[
            1,
            3,
            self.input_size.1 as usize,
            self.input_size.0 as usize,
        ]));

        // Apply ImageNet normalization (RF-DETR)
        for y in 0..self.input_size.1 as usize {
            for x in 0..self.input_size.0 as usize {
                let pixel_idx = (y * self.input_size.0 as usize + x) * 3;

                // Normalize to [0, 1] range then apply ImageNet normalization
                let r = self.letterboxed_buffer[pixel_idx] as f32 / 255.0;
                let g = self.letterboxed_buffer[pixel_idx + 1] as f32 / 255.0;
                let b = self.letterboxed_buffer[pixel_idx + 2] as f32 / 255.0;

                input[[0, 0, y, x]] = (r - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
                input[[0, 1, y, x]] = (g - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
                input[[0, 2, y, x]] = (b - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
            }
        }

        Ok((input, scale, offset_x as f32, offset_y as f32))
    }
}

impl Default for PreProcessor {
    fn default() -> Self {
        Self::new(DEFAULT_INPUT_SIZE)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use flatbuffers::FlatBufferBuilder;

    /// Helper function to create a FlatBuffers Frame for testing
    fn create_test_frame(
        width: u32,
        height: u32,
        format: bridge::ColorFormat,
        pixels: Vec<u8>,
    ) -> Vec<u8> {
        let mut builder = FlatBufferBuilder::new();

        let pixel_vector = builder.create_vector(&pixels);

        let frame = schema::Frame::create(
            &mut builder,
            &schema::FrameArgs {
                frame_number: 1,
                timestamp_ns: 0,
                camera_id: 0,
                width,
                height,
                channels: 3,
                format,
                pixels: Some(pixel_vector),
            },
        );

        builder.finish(frame, None);
        builder.finished_data().to_vec()
    }

    /// Test BGR to RGB conversion
    #[test]
    fn test_bgr_to_rgb_conversion() {
        let pixels = vec![
            255, 0, 0, // Blue pixel (BGR: 255,0,0 → RGB: 0,0,255)
            0, 255, 0, // Green pixel (BGR: 0,255,0 → RGB: 0,255,0)
            0, 0, 255, // Red pixel (BGR: 0,0,255 → RGB: 255,0,0)
            128, 128, 128, // Gray pixel
        ];

        let frame_data = create_test_frame(2, 2, bridge::ColorFormat::BGR, pixels);
        let frame = flatbuffers::root::<schema::Frame>(&frame_data).unwrap();

        let mut preprocessor = PreProcessor::default();
        let (output, scale, offset_x, offset_y) = preprocessor
            .preprocess_frame(
                frame.pixels().unwrap(),
                frame.width(),
                frame.height(),
                frame.format(),
            )
            .unwrap();

        // Verify output shape (512x512 for RF-DETR)
        assert_eq!(output.shape(), &[1, 3, 512, 512]);

        // Verify scale and offsets
        // 2x2 → 512x512: scale = 512/2 = 256
        assert_eq!(scale, 256.0);
        assert_eq!(offset_x, 0.0);
        assert_eq!(offset_y, 0.0);

        // Validate basic shape
        assert!(output.shape()[0] == 1);
        assert!(output.shape()[1] == 3);
    }

    /// Test RGB passthrough
    #[test]
    fn test_rgb_to_rgb_passthrough() {
        let pixels = vec![
            255, 0, 0, // Red pixel
            0, 255, 0, // Green pixel
            0, 0, 255, // Blue pixel
            255, 255, 255, // White pixel
        ];

        let frame_data = create_test_frame(2, 2, bridge::ColorFormat::RGB, pixels);
        let frame = flatbuffers::root::<schema::Frame>(&frame_data).unwrap();

        let mut preprocessor = PreProcessor::default();
        let result = preprocessor.preprocess_frame(
            frame.pixels().unwrap(),
            frame.width(),
            frame.height(),
            frame.format(),
        );

        assert!(result.is_ok(), "RGB passthrough should succeed");
        let (output, _, _, _) = result.unwrap();
        assert_eq!(output.shape(), &[1, 3, 512, 512]);
    }

    /// Test that grayscale format returns error
    #[test]
    fn test_gray_format_returns_error() {
        let pixels = vec![128; 100]; // 10x10 grayscale

        let frame_data = create_test_frame(10, 10, bridge::ColorFormat::GRAY, pixels);
        let frame = flatbuffers::root::<schema::Frame>(&frame_data).unwrap();

        let mut preprocessor = PreProcessor::default();
        let result = preprocessor.preprocess_frame(
            frame.pixels().unwrap(),
            frame.width(),
            frame.height(),
            frame.format(),
        );

        assert!(result.is_err(), "Grayscale should return error");
        assert!(
            result.unwrap_err().to_string().contains("Grayscale"),
            "Error should mention grayscale"
        );
    }

    /// Test buffer size mismatch detection
    #[test]
    fn test_buffer_size_mismatch_detection() {
        let pixels = vec![0u8; 200]; // Wrong size for 10x10

        let frame_data = create_test_frame(10, 10, bridge::ColorFormat::RGB, pixels);
        let frame = flatbuffers::root::<schema::Frame>(&frame_data).unwrap();

        let mut preprocessor = PreProcessor::default();
        let result = preprocessor.preprocess_frame(
            frame.pixels().unwrap(),
            frame.width(),
            frame.height(),
            frame.format(),
        );

        assert!(result.is_err(), "Size mismatch should return error");
        assert!(
            result.unwrap_err().to_string().contains("mismatch"),
            "Error should mention mismatch"
        );
    }

    /// Test letterboxing preserves aspect ratio
    #[test]
    fn test_letterboxing_preserves_aspect_ratio() {
        // 800x600 image (4:3 aspect ratio)
        let pixels = vec![128u8; 800 * 600 * 3];

        let frame_data = create_test_frame(800, 600, bridge::ColorFormat::RGB, pixels);
        let frame = flatbuffers::root::<schema::Frame>(&frame_data).unwrap();

        let mut preprocessor = PreProcessor::default();
        let (output, scale, offset_x, offset_y) = preprocessor
            .preprocess_frame(
                frame.pixels().unwrap(),
                frame.width(),
                frame.height(),
                frame.format(),
            )
            .unwrap();

        // Scale should be min(512/800, 512/600) = 512/800 = 0.64
        assert_eq!(scale, 0.64, "Scale should preserve aspect ratio");

        // Resized dimensions: 800*0.64 = 512, 600*0.64 = 384
        // Offset X: (512 - 512) / 2 = 0
        // Offset Y: (512 - 384) / 2 = 64
        assert_eq!(offset_x, 0.0, "X offset should be 0 for wide image");
        assert_eq!(offset_y, 64.0, "Y offset should center vertically");

        // Output shape should always be 512x512
        assert_eq!(output.shape(), &[1, 3, 512, 512]);
    }

    /// Test ImageNet normalization is applied
    #[test]
    fn test_imagenet_normalization() {
        // Create image with known pixel values (128, 128, 128 = mid gray)
        let pixels = vec![128u8; 2 * 2 * 3];

        let frame_data = create_test_frame(2, 2, bridge::ColorFormat::RGB, pixels);
        let frame = flatbuffers::root::<schema::Frame>(&frame_data).unwrap();

        let mut preprocessor = PreProcessor::new((512, 512));
        let (output, _, _, _) = preprocessor
            .preprocess_frame(
                frame.pixels().unwrap(),
                frame.width(),
                frame.height(),
                frame.format(),
            )
            .unwrap();

        // Verify output shape is 512x512
        assert_eq!(output.shape(), &[1, 3, 512, 512]);

        // For gray 128 (0.502) with ImageNet norm:
        //   R: (0.502 - 0.485) / 0.229 ≈ 0.074
        //   G: (0.502 - 0.456) / 0.224 ≈ 0.205
        //   B: (0.502 - 0.406) / 0.225 ≈ 0.427
        // Channels should have different values

        let r = output[[0, 0, 256, 256]];
        let g = output[[0, 1, 256, 256]];
        let b = output[[0, 2, 256, 256]];

        // After ImageNet normalization, channels should differ
        assert!(
            (r - g).abs() > 0.1,
            "R and G should differ with ImageNet norm (R={}, G={})",
            r,
            g
        );
        assert!(
            (g - b).abs() > 0.1,
            "G and B should differ with ImageNet norm (G={}, B={})",
            g,
            b
        );

        // Check approximate expected values
        assert!(
            (r - 0.074).abs() < 0.1,
            "R channel should be ~0.074 (got {})",
            r
        );
        assert!(
            (g - 0.205).abs() < 0.1,
            "G channel should be ~0.205 (got {})",
            g
        );
        assert!(
            (b - 0.427).abs() < 0.1,
            "B channel should be ~0.427 (got {})",
            b
        );
    }
}
