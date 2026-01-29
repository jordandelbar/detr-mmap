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
use common::span;
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

/// PTX kernel embedded at compile time (compiled by nvcc in build.rs)
const PREPROCESS_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/preprocess.ptx"));

/// GPU-accelerated image preprocessor
pub struct GpuPreProcessor {
    /// Target input size (width, height)
    input_size: (u32, u32),
    /// CUDA device handle
    device: Arc<CudaDevice>,
    /// Pre-allocated device buffer for input image (RGB u8)
    /// Size matches the last processed frame; reallocated if frame size changes
    d_input: CudaSlice<u8>,
    /// Current input buffer size in pixels (width * height)
    current_input_pixels: usize,
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
        let device = CudaDevice::new(device_id).context("Failed to initialize CUDA device")?;

        // Load the PTX kernel
        let ptx = Ptx::from_src(PREPROCESS_PTX);
        device
            .load_ptx(ptx, "preprocess", &["preprocess_kernel"])
            .context("Failed to load preprocess PTX")?;

        let max_input_pixels = (max_input_size.0 * max_input_size.1) as usize;
        let output_pixels = (input_size.0 * input_size.1) as usize;

        // Pre-allocate device buffers
        // Start with a small input buffer; it will be reallocated on first use
        let d_input = device
            .alloc_zeros::<u8>(3)
            .context("Failed to allocate input buffer")?;

        let d_output = device
            .alloc_zeros::<f32>(output_pixels * 3)
            .context("Failed to allocate output buffer")?;

        Ok(Self {
            input_size,
            device,
            d_input,
            current_input_pixels: 0,
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

    /// Copy the output buffer from device to host (for testing/verification)
    pub fn copy_output_to_host(&self) -> Result<Vec<f32>> {
        self.device
            .dtoh_sync_copy(&self.d_output)
            .context("Failed to copy output from device")
    }

    /// Upload pixels to device memory (for benchmarking kernel-only performance)
    ///
    /// Call this once to upload data, then use `run_kernel` to benchmark just the kernel.
    pub fn upload_to_device(&mut self, pixels: &[u8], width: u32, height: u32) -> Result<()> {
        let _s = span!("host_to_device_transfer");

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

        // Reallocate input buffer if frame size changed
        if num_pixels != self.current_input_pixels {
            self.d_input = self
                .device
                .alloc_zeros::<u8>(input_bytes)
                .context("Failed to reallocate input buffer")?;
            self.current_input_pixels = num_pixels;
        }

        // Copy input to device
        self.device
            .htod_copy_into(pixels.to_vec(), &mut self.d_input)
            .context("Failed to copy input to device")?;

        Ok(())
    }

    /// Run the preprocessing kernel only (assumes data already uploaded via `upload_to_device`)
    ///
    /// This is useful for benchmarking kernel performance without host-to-device copy overhead.
    pub fn run_kernel(&mut self, width: u32, height: u32) -> Result<(u64, f32, f32, f32)> {
        let _s = span!("preprocess_kernel");

        // Calculate letterbox parameters
        let scale =
            (self.input_size.0 as f32 / width as f32).min(self.input_size.1 as f32 / height as f32);
        let new_width = (width as f32 * scale) as u32;
        let new_height = (height as f32 * scale) as u32;
        let offset_x = (self.input_size.0 - new_width) / 2;
        let offset_y = (self.input_size.1 - new_height) / 2;

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

        // Kernel parameters (11 params - ImageNet constants are embedded in kernel)
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
                ),
            )
            .context("Failed to launch preprocess kernel")?;
        }

        // Synchronize to ensure kernel completion
        self.device.synchronize().context("Failed to synchronize")?;

        Ok((
            self.output_device_ptr(),
            scale,
            offset_x as f32,
            offset_y as f32,
        ))
    }

    /// Preprocess an image on the GPU (full pipeline: upload + kernel)
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
        self.upload_to_device(pixels, width, height)?;
        self.run_kernel(width, height)
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
    use crate::CpuPreProcessor;

    /// Helper to check if GPU preprocessing is fully working
    /// Returns None if working, Some(reason) if not
    fn gpu_not_available() -> Option<String> {
        match GpuPreProcessor::new((64, 64), (128, 128)) {
            Ok(mut gpu) => {
                // Try a simple preprocess to verify kernel works
                let test_pixels = vec![128u8; 128 * 128 * 3];
                match gpu.preprocess_to_device(&test_pixels, 128, 128) {
                    Ok(_) => None,
                    Err(e) => Some(format!("Kernel execution failed: {}", e)),
                }
            }
            Err(e) => Some(format!("GPU init failed: {}", e)),
        }
    }

    #[test]
    fn test_gpu_preprocessor_creation() {
        let result = GpuPreProcessor::new((512, 512), (1920, 1080));
        match result {
            Ok(_) => eprintln!("GPU preprocessor created successfully"),
            Err(e) => eprintln!(
                "GPU preprocessor creation failed (expected if no CUDA): {:?}",
                e
            ),
        }
    }

    #[test]
    fn test_gpu_vs_cpu_preprocessing() {
        if let Some(reason) = gpu_not_available() {
            eprintln!("Skipping GPU vs CPU test: {}", reason);
            return;
        }

        let input_size = (512, 512);
        let width = 640u32;
        let height = 480u32;

        // Create test image (gradient pattern for better comparison)
        let mut pixels = vec![0u8; (width * height * 3) as usize];
        for y in 0..height {
            for x in 0..width {
                let idx = ((y * width + x) * 3) as usize;
                pixels[idx] = (x % 256) as u8; // R
                pixels[idx + 1] = (y % 256) as u8; // G
                pixels[idx + 2] = ((x + y) % 256) as u8; // B
            }
        }

        // CPU preprocessing
        let mut cpu = CpuPreProcessor::new(input_size);
        let (cpu_output, cpu_scale, cpu_offset_x, cpu_offset_y) = cpu
            .preprocess_from_u8_slice(&pixels, width, height)
            .unwrap();

        // GPU preprocessing
        let mut gpu = GpuPreProcessor::new(input_size, (width, height)).unwrap();
        let (_, gpu_scale, gpu_offset_x, gpu_offset_y) =
            gpu.preprocess_to_device(&pixels, width, height).unwrap();
        let gpu_output = gpu.copy_output_to_host().unwrap();

        // Verify transformation parameters match
        assert_eq!(cpu_scale, gpu_scale, "Scale mismatch");
        assert_eq!(cpu_offset_x, gpu_offset_x, "Offset X mismatch");
        assert_eq!(cpu_offset_y, gpu_offset_y, "Offset Y mismatch");

        // Verify output shapes match
        let cpu_flat = cpu_output.as_slice().unwrap();
        assert_eq!(
            cpu_flat.len(),
            gpu_output.len(),
            "Output size mismatch: CPU {} vs GPU {}",
            cpu_flat.len(),
            gpu_output.len()
        );

        // Compare outputs with tolerance (bilinear interpolation may differ slightly)
        let tolerance = 0.05; // Allow 5% tolerance for interpolation differences
        let mut max_diff = 0.0f32;
        let mut diff_count = 0;
        let total_pixels = cpu_flat.len();

        for (i, (cpu_val, gpu_val)) in cpu_flat.iter().zip(gpu_output.iter()).enumerate() {
            let diff = (cpu_val - gpu_val).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            if diff > tolerance {
                diff_count += 1;
                if diff_count <= 10 {
                    eprintln!(
                        "Diff at index {}: CPU={:.6}, GPU={:.6}, diff={:.6}",
                        i, cpu_val, gpu_val, diff
                    );
                }
            }
        }

        let diff_ratio = diff_count as f64 / total_pixels as f64;
        eprintln!(
            "GPU vs CPU comparison: max_diff={:.6}, diff_count={}/{} ({:.2}%)",
            max_diff,
            diff_count,
            total_pixels,
            diff_ratio * 100.0
        );

        // Allow up to 1% of pixels to differ beyond tolerance
        assert!(
            diff_ratio < 0.01,
            "Too many pixels differ: {:.2}% (max allowed 1%)",
            diff_ratio * 100.0
        );

        // Max difference should be reasonable
        assert!(
            max_diff < 0.5,
            "Maximum difference too large: {:.6}",
            max_diff
        );

        eprintln!("GPU vs CPU test passed!");
    }

    #[test]
    fn test_gpu_letterbox_padding() {
        if let Some(reason) = gpu_not_available() {
            eprintln!("Skipping GPU letterbox test: {}", reason);
            return;
        }

        let input_size = (512, 512);
        // Wide image (will have vertical padding)
        let width = 800u32;
        let height = 400u32;

        // Create solid red image
        let pixels = vec![255u8, 0, 0].repeat((width * height) as usize);

        let mut gpu = GpuPreProcessor::new(input_size, (width, height)).unwrap();
        let (_, scale, offset_x, offset_y) =
            gpu.preprocess_to_device(&pixels, width, height).unwrap();
        let output = gpu.copy_output_to_host().unwrap();

        // Verify letterbox parameters
        let expected_scale = 512.0 / 800.0; // 0.64
        assert!(
            (scale - expected_scale).abs() < 0.01,
            "Scale should be ~{}, got {}",
            expected_scale,
            scale
        );
        assert_eq!(offset_x, 0.0, "X offset should be 0 for wide image");
        assert!(offset_y > 0.0, "Y offset should be positive for wide image");

        // Check that padding region has letterbox gray value (114)
        // After normalization: (114/255 - mean) / std
        let gray_norm = 114.0 / 255.0;
        let expected_r = (gray_norm - 0.485) / 0.229;
        let expected_g = (gray_norm - 0.456) / 0.224;
        let expected_b = (gray_norm - 0.406) / 0.225;

        // Check a pixel in the top padding region (y=0, x=256)
        let spatial = (input_size.0 * input_size.1) as usize;
        let idx = 256; // Top-center pixel
        let r = output[idx];
        let g = output[idx + spatial];
        let b = output[idx + 2 * spatial];

        assert!(
            (r - expected_r).abs() < 0.1,
            "Padding R channel should be ~{:.3}, got {:.3}",
            expected_r,
            r
        );
        assert!(
            (g - expected_g).abs() < 0.1,
            "Padding G channel should be ~{:.3}, got {:.3}",
            expected_g,
            g
        );
        assert!(
            (b - expected_b).abs() < 0.1,
            "Padding B channel should be ~{:.3}, got {:.3}",
            expected_b,
            b
        );

        eprintln!("GPU letterbox padding test passed!");
    }
}
