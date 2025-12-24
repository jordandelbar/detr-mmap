use ndarray::{Array, IxDyn};

pub mod ort;

pub trait InferenceBackend {
    fn load_model(path: &str) -> anyhow::Result<Self>
    where
        Self: Sized;
    fn infer(
        &mut self,
        images: &Array<f32, IxDyn>,
        orig_sizes: &Array<i64, IxDyn>,
    ) -> anyhow::Result<InferenceOutput>;
}

pub struct InferenceOutput {
    pub labels: ndarray::ArrayD<i64>,
    pub boxes: ndarray::ArrayD<f32>,
    pub scores: ndarray::ArrayD<f32>,
}
