use ndarray::{Array, IxDyn};

#[cfg(feature = "ort-backend")]
pub mod ort;

#[cfg(feature = "trt-backend")]
pub mod trt;

pub trait InferenceBackend {
    fn load_model(path: &str) -> anyhow::Result<Self>
    where
        Self: Sized;
    fn infer(
        &mut self,
        images: &Array<f32, IxDyn>,
    ) -> anyhow::Result<InferenceOutput>;
}

/// RF-DETR inference output
pub struct InferenceOutput {
    pub dets: ndarray::ArrayD<f32>,    // [1, 300, 4] cxcywh (normalized 0-1)
    pub logits: ndarray::ArrayD<f32>,  // [1, 300, num_classes] class logits
}
