pub mod config;
pub mod cpu;
#[cfg(feature = "cuda")]
pub mod gpu;

use ndarray::{Array, IxDyn};

pub use config::DEFAULT_INPUT_SIZE;
pub use cpu::CpuPreProcessor;
#[cfg(feature = "cuda")]
pub use gpu::GpuPreProcessor;

/// Output from preprocessing - either CPU array or GPU device pointer
#[derive(Debug)]
pub enum PreprocessOutput {
    /// CPU array ready for host-side inference
    Cpu(Array<f32, IxDyn>),
    /// GPU device pointer with length (no copy needed for TRT)
    Gpu {
        /// Device pointer to the preprocessed data
        ptr: u64,
        /// Number of elements (not bytes)
        len: usize,
    },
}

/// Result of preprocessing including transformation parameters
#[derive(Debug)]
pub struct PreprocessResult {
    /// Preprocessed image data (CPU or GPU)
    pub data: PreprocessOutput,
    /// Scale factor applied during letterboxing
    pub scale: f32,
    /// X offset from letterboxing (in pixels)
    pub offset_x: f32,
    /// Y offset from letterboxing (in pixels)
    pub offset_y: f32,
}

/// Trait for image preprocessing implementations
pub trait Preprocess {
    /// Preprocess an image for inference
    ///
    /// # Arguments
    /// * `pixels` - RGB pixel data in HWC format
    /// * `width` - Image width
    /// * `height` - Image height
    ///
    /// # Returns
    /// Preprocessed result with transformation parameters
    fn preprocess(
        &mut self,
        pixels: &[u8],
        width: u32,
        height: u32,
    ) -> anyhow::Result<PreprocessResult>;

    /// Get the input size this preprocessor targets
    fn input_size(&self) -> (u32, u32);
}

// Re-export the old PreProcessor name for backwards compatibility
pub use cpu::CpuPreProcessor as PreProcessor;
