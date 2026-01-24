use ndarray::{Array, IxDyn};
use preprocess::PreprocessOutput;

#[cfg(feature = "ort-backend")]
pub mod ort;

#[cfg(feature = "trt-backend")]
pub mod trt;

pub trait InferenceBackend {
    fn load_model(path: &str) -> anyhow::Result<Self>
    where
        Self: Sized;

    /// Run inference with a CPU array input
    fn infer(&mut self, images: &Array<f32, IxDyn>) -> anyhow::Result<InferenceOutput>;

    /// Run inference with preprocessed input (CPU or GPU)
    ///
    /// Default implementation handles CPU input by delegating to `infer()`.
    /// GPU input returns an error unless the backend overrides this method.
    fn infer_preprocessed(&mut self, input: &PreprocessOutput) -> anyhow::Result<InferenceOutput> {
        match input {
            PreprocessOutput::Cpu(arr) => self.infer(arr),
            PreprocessOutput::Gpu { .. } => {
                anyhow::bail!("GPU input not supported by this backend")
            }
        }
    }
}

pub struct InferenceOutput {
    pub dets: ndarray::ArrayD<f32>, // [1, 300, 4] cxcywh (normalized 0-1)
    pub logits: ndarray::ArrayD<f32>, // [1, 300, num_classes] class logits
}
