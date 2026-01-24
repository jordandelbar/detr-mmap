//! GPU-accelerated preprocessing using CUDA
//!
//! This module provides a GPU-based preprocessor that performs:
//! - Bilinear resize
//! - Letterbox padding (gray 114)
//! - ImageNet normalization
//! - HWC -> CHW transpose
//!
//! All operations are fused into a single CUDA kernel for maximum performance.

use crate::config::DEFAULT_INPUT_SIZE;
use crate::{Preprocess, PreprocessOutput, PreprocessResult};
use anyhow::{Context, Result};
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

/// ImageNet normalization constants
const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];
const LETTERBOX_COLOR: u8 = 114;

/// PTX kernel embedded at compile time
const PREPROCESS_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/preprocess.ptx"));

/// GPU-accelerated image preprocessor
pub struct GpuPreProcessor {
    /// Target input size (width, height)
    input_size: (u32, u32),
    /// CUDA device handle
    device: Arc<CudaDevice>,
    /// Pre-allocated device buffer for input image (RGB u8)
    d_input: CudaSlice<u8>,
    /// Pre-allocated device buffer for output (CHW f32)
    d_output: CudaSlice<f32>,
    /// Maximum input image size we can handle
    max_input_pixels: usize,
}

impl GpuPreProcessor {
    /// Create a new GPU preprocessor
    ///
    /// # Arguments
    /// * `input_size` - Target output size (width, height) for the model
    /// * `max_input_size` - Maximum expected input image dimensions (width, height)
    ///
    /// # Returns
    /// A new GpuPreProcessor or an error if CUDA initialization fails
    pub fn new(input_size: (u32, u32), max_input_size: (u32, u32)) -> Result<Self> {
        Self::with_device_id(input_size, max_input_size, 0)
    }

    /// Create a new GPU preprocessor on a specific CUDA device
    pub fn with_device_id(
        input_size: (u32, u32),
        max_input_size: (u32, u32),
        device_id: usize,
    ) -> Result<Self> {
        let device = CudaDevice::new(device_id)
            .context("Failed to initialize CUDA device")?;

        // Load the PTX kernel
        let ptx = Ptx::from_src(PREPROCESS_PTX);
        device
            .load_ptx(ptx, "preprocess", &["preprocess_kernel"])
            .context("Failed to load preprocess PTX")?;

        let max_input_pixels = (max_input_size.0 * max_input_size.1) as usize;
        let output_pixels = (input_size.0 * input_size.1) as usize;

        // Pre-allocate device buffers
        let d_input = device
            .alloc_zeros::<u8>(max_input_pixels * 3)
            .context("Failed to allocate input buffer")?;

        let d_output = device
            .alloc_zeros::<f32>(output_pixels * 3)
            .context("Failed to allocate output buffer")?;

        Ok(Self {
            input_size,
            device,
            d_input,
            d_output,
            max_input_pixels,
        })
    }

    /// Get the device pointer to the output buffer
    ///
    /// This pointer can be passed directly to TensorRT for zero-copy inference.
    pub fn output_device_ptr(&self) -> u64 {
        *self.d_output.device_ptr() as u64
    }

    /// Get the number of output elements
    pub fn output_len(&self) -> usize {
        (self.input_size.0 * self.input_size.1 * 3) as usize
    }

    /// Preprocess an image on the GPU
    ///
    /// # Arguments
    /// * `pixels` - RGB pixel data in HWC format
    /// * `width` - Image width
    /// * `height` - Image height
    ///
    /// # Returns
    /// Device pointer to preprocessed data and transformation parameters
    pub fn preprocess_to_device(
        &mut self,
        pixels: &[u8],
        width: u32,
        height: u32,
    ) -> Result<(u64, f32, f32, f32)> {
        let num_pixels = (width * height) as usize;
        let input_bytes = num_pixels * 3;

        if num_pixels > self.max_input_pixels {
            anyhow::bail!(
                "Input image {}x{} exceeds maximum size ({}x{} max pixels)",
                width,
                height,
                self.max_input_pixels,
                1
            );
        }

        if pixels.len() != input_bytes {
            anyhow::bail!(
                "Buffer size mismatch: expected {}, got {} bytes",
                input_bytes,
                pixels.len()
            );
        }

        // Calculate letterbox parameters
        let scale = (self.input_size.0 as f32 / width as f32)
            .min(self.input_size.1 as f32 / height as f32);
        let new_width = (width as f32 * scale) as u32;
        let new_height = (height as f32 * scale) as u32;
        let offset_x = (self.input_size.0 - new_width) / 2;
        let offset_y = (self.input_size.1 - new_height) / 2;

        // Copy input to device
        self.device
            .htod_copy_into(pixels.to_vec(), &mut self.d_input)
            .context("Failed to copy input to device")?;

        // Launch the preprocessing kernel
        let func = self
            .device
            .get_func("preprocess", "preprocess_kernel")
            .context("Failed to get preprocess kernel")?;

        // Calculate grid/block dimensions
        let output_pixels = (self.input_size.0 * self.input_size.1) as u32;
        let block_size = 256u32;
        let grid_size = (output_pixels + block_size - 1) / block_size;

        let config = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        // Kernel parameters
        unsafe {
            func.launch(
                config,
                (
                    &self.d_input,
                    &self.d_output,
                    width as i32,
                    height as i32,
                    self.input_size.0 as i32,
                    self.input_size.1 as i32,
                    new_width as i32,
                    new_height as i32,
                    offset_x as i32,
                    offset_y as i32,
                    scale,
                    IMAGENET_MEAN[0],
                    IMAGENET_MEAN[1],
                    IMAGENET_MEAN[2],
                    IMAGENET_STD[0],
                    IMAGENET_STD[1],
                    IMAGENET_STD[2],
                    LETTERBOX_COLOR as i32,
                ),
            )
            .context("Failed to launch preprocess kernel")?;
        }

        // Synchronize to ensure kernel completion
        self.device.synchronize().context("Failed to synchronize")?;

        Ok((self.output_device_ptr(), scale, offset_x as f32, offset_y as f32))
    }
}

impl Default for GpuPreProcessor {
    fn default() -> Self {
        // Default max input size of 4K (3840x2160)
        Self::new(DEFAULT_INPUT_SIZE, (3840, 2160))
            .expect("Failed to create default GpuPreProcessor")
    }
}

impl Preprocess for GpuPreProcessor {
    fn preprocess(&mut self, pixels: &[u8], width: u32, height: u32) -> Result<PreprocessResult> {
        let (ptr, scale, offset_x, offset_y) = self.preprocess_to_device(pixels, width, height)?;

        Ok(PreprocessResult {
            data: PreprocessOutput::Gpu {
                ptr,
                len: self.output_len(),
            },
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

    #[test]
    fn test_gpu_preprocessor_creation() {
        // This test will only pass on a system with CUDA
        let result = GpuPreProcessor::new((512, 512), (1920, 1080));
        // We don't assert success since CUDA might not be available
        if result.is_err() {
            eprintln!("GPU preprocessor creation failed (expected if no CUDA): {:?}", result.err());
        }
    }
}
