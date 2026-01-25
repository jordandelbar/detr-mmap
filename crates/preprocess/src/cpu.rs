use crate::config::DEFAULT_INPUT_SIZE;
use crate::{Preprocess, PreprocessOutput, PreprocessResult};
use common::span;
use fast_image_resize::{
    FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer,
    images::{Image, ImageRef},
};
use ndarray::{Array, IxDyn};
use std::default::Default;

const LETTERBOX_COLOR: u8 = 114;
const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

pub struct CpuPreProcessor {
    pub input_size: (u32, u32),
    letterboxed_buffer: Vec<u8>,
}

impl CpuPreProcessor {
    pub fn new(input_size: (u32, u32)) -> Self {
        Self {
            input_size,
            letterboxed_buffer: vec![LETTERBOX_COLOR; (input_size.0 * input_size.1 * 3) as usize],
        }
    }

    pub fn preprocess_frame(
        &mut self,
        pixels: flatbuffers::Vector<u8>,
        width: u32,
        height: u32,
    ) -> anyhow::Result<(Array<f32, IxDyn>, f32, f32, f32)> {
        let _s = span!("preprocess_frame");

        tracing::trace!(
            width,
            height,
            pixel_bytes = pixels.len(),
            "Preprocessing frame dimensions"
        );

        let expected_size = (width * height * 3) as usize;
        if pixels.len() != expected_size {
            anyhow::bail!(
                "Buffer size mismatch: expected {}, got {} bytes",
                expected_size,
                pixels.len()
            );
        }

        let (scale, offset_x, offset_y, resized) =
            self.resize_and_letterbox(pixels.bytes(), width, height)?;

        let input = Self::normalize(&resized)?;

        Ok((input, scale, offset_x, offset_y))
    }

    pub fn preprocess_from_u8_slice(
        &mut self,
        pixels: &[u8],
        width: u32,
        height: u32,
    ) -> anyhow::Result<(Array<f32, IxDyn>, f32, f32, f32)> {
        let (scale, offset_x, offset_y, resized) =
            self.resize_and_letterbox(pixels, width, height)?;
        let input = Self::normalize(&resized)?;
        Ok((input, scale, offset_x, offset_y))
    }

    fn resize_and_letterbox(
        &mut self,
        pixels: &[u8],
        width: u32,
        height: u32,
    ) -> anyhow::Result<(f32, f32, f32, Image<'_>)> {
        let _s = span!("resize_and_letterbox");

        let scale =
            (self.input_size.0 as f32 / width as f32).min(self.input_size.1 as f32 / height as f32);
        let new_width = (width as f32 * scale) as u32;
        let new_height = (height as f32 * scale) as u32;

        let offset_x = (self.input_size.0 - new_width) / 2;
        let offset_y = (self.input_size.1 - new_height) / 2;

        let src = ImageRef::new(width, height, pixels, PixelType::U8x3)?;

        let mut resized = Image::new(new_width, new_height, PixelType::U8x3);

        Resizer::new().resize(
            &src,
            &mut resized,
            &ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Bilinear)),
        )?;

        self.letterboxed_buffer.fill(LETTERBOX_COLOR);

        let resized_data = resized.buffer();
        let stride = self.input_size.0 * 3;

        for y in 0..new_height {
            let src_row = (y * new_width * 3) as usize;
            let dst_row = ((y + offset_y) * stride + offset_x * 3) as usize;

            self.letterboxed_buffer[dst_row..dst_row + (new_width * 3) as usize]
                .copy_from_slice(&resized_data[src_row..src_row + (new_width * 3) as usize]);
        }

        let final_img = Image::from_slice_u8(
            self.input_size.0,
            self.input_size.1,
            &mut self.letterboxed_buffer,
            PixelType::U8x3,
        )?;

        Ok((scale, offset_x as f32, offset_y as f32, final_img))
    }

    fn normalize(image: &Image) -> anyhow::Result<Array<f32, IxDyn>> {
        let _s = span!("normalize");

        let width = image.width() as usize;
        let height = image.height() as usize;
        let spatial = width * height;

        let mut output = vec![0.0f32; 3 * spatial];
        let buf = image.buffer();

        for (i, px) in buf.chunks_exact(3).enumerate() {
            let r = px[0] as f32 / 255.0;
            let g = px[1] as f32 / 255.0;
            let b = px[2] as f32 / 255.0;

            output[i] = (r - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
            output[i + spatial] = (g - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
            output[i + 2 * spatial] = (b - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
        }

        Ok(Array::from_shape_vec(
            IxDyn(&[1, 3, height, width]),
            output,
        )?)
    }
}

impl Default for CpuPreProcessor {
    fn default() -> Self {
        Self::new(DEFAULT_INPUT_SIZE)
    }
}

impl Preprocess for CpuPreProcessor {
    fn preprocess(
        &mut self,
        pixels: &[u8],
        width: u32,
        height: u32,
    ) -> anyhow::Result<PreprocessResult> {
        let (array, scale, offset_x, offset_y) =
            self.preprocess_from_u8_slice(pixels, width, height)?;
        Ok(PreprocessResult {
            data: PreprocessOutput::Cpu(array),
            scale,
            offset_x,
            offset_y,
        })
    }

    fn input_size(&self) -> (u32, u32) {
        self.input_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use flatbuffers::FlatBufferBuilder;

    /// Helper function to create a FlatBuffers Frame for testing
    fn create_test_frame(width: u32, height: u32, pixels: Vec<u8>) -> Vec<u8> {
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
                pixels: Some(pixel_vector),
                trace: None,
            },
        );

        builder.finish(frame, None);
        builder.finished_data().to_vec()
    }

    /// Test RGB preprocessing
    #[test]
    fn test_rgb_preprocessing() {
        let pixels = vec![
            255, 0, 0, // Red pixel
            0, 255, 0, // Green pixel
            0, 0, 255, // Blue pixel
            255, 255, 255, // White pixel
        ];

        let frame_data = create_test_frame(2, 2, pixels);
        let frame = flatbuffers::root::<schema::Frame>(&frame_data).unwrap();

        let mut preprocessor = CpuPreProcessor::default();
        let result =
            preprocessor.preprocess_frame(frame.pixels().unwrap(), frame.width(), frame.height());

        assert!(result.is_ok(), "RGB preprocessing should succeed");
        let (output, _, _, _) = result.unwrap();
        assert_eq!(output.shape(), &[1, 3, 512, 512]);
    }

    /// Test buffer size mismatch detection
    #[test]
    fn test_buffer_size_mismatch_detection() {
        let pixels = vec![0u8; 200]; // Wrong size for 10x10

        let frame_data = create_test_frame(10, 10, pixels);
        let frame = flatbuffers::root::<schema::Frame>(&frame_data).unwrap();

        let mut preprocessor = CpuPreProcessor::default();
        let result =
            preprocessor.preprocess_frame(frame.pixels().unwrap(), frame.width(), frame.height());

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

        let frame_data = create_test_frame(800, 600, pixels);
        let frame = flatbuffers::root::<schema::Frame>(&frame_data).unwrap();

        let mut preprocessor = CpuPreProcessor::default();
        let (output, scale, offset_x, offset_y) = preprocessor
            .preprocess_frame(frame.pixels().unwrap(), frame.width(), frame.height())
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

        let frame_data = create_test_frame(2, 2, pixels);
        let frame = flatbuffers::root::<schema::Frame>(&frame_data).unwrap();

        let mut preprocessor = CpuPreProcessor::new((512, 512));
        let (output, _, _, _) = preprocessor
            .preprocess_frame(frame.pixels().unwrap(), frame.width(), frame.height())
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

    /// Test the Preprocess trait implementation
    #[test]
    fn test_preprocess_trait() {
        let pixels = vec![128u8; 100 * 100 * 3];
        let mut preprocessor = CpuPreProcessor::default();

        let result = preprocessor.preprocess(&pixels, 100, 100);
        assert!(result.is_ok());

        let preprocess_result = result.unwrap();
        assert!(matches!(preprocess_result.data, PreprocessOutput::Cpu(_)));
        assert!(preprocess_result.scale > 0.0);
    }
}
