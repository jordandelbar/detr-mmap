use super::{InferenceBackend, InferenceOutput};
use ndarray::{Array, IxDyn};
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::TensorRef,
};

pub struct OrtBackend {
    session: Session,
}

impl InferenceBackend for OrtBackend {
    fn load_model(path: &str) -> anyhow::Result<Self> {
        // Try TensorRT first (with CUDA fallback), then fall back to CPU if unavailable
        let session = match Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .with_execution_providers([
                ort::execution_providers::TensorRTExecutionProvider::default()
                    .with_device_id(0)
                    .build(),
                ort::execution_providers::CUDAExecutionProvider::default()
                    .with_device_id(0)
                    .build(),
            ])?
            .commit_from_file(path)
        {
            Ok(session) => {
                tracing::info!("Model loaded with TensorRT/CUDA execution provider");
                session
            }
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "Failed to load model with TensorRT/CUDA, falling back to CPU"
                );
                Session::builder()?
                    .with_optimization_level(GraphOptimizationLevel::Level3)?
                    .with_intra_threads(4)?
                    .commit_from_file(path)?
            }
        };

        Ok(Self { session })
    }

    fn infer(
        &mut self,
        images: &Array<f32, IxDyn>,
        orig_sizes: &Array<i64, IxDyn>,
    ) -> anyhow::Result<InferenceOutput> {
        let outputs = self.session.run(ort::inputs![
            "images" => TensorRef::from_array_view(images)?,
            "orig_target_sizes" => TensorRef::from_array_view(orig_sizes)?
        ])?;

        let labels = outputs["labels"].try_extract_array::<i64>()?;
        let boxes = outputs["boxes"].try_extract_array::<f32>()?;
        let scores = outputs["scores"].try_extract_array::<f32>()?;

        Ok(InferenceOutput {
            labels: labels.into_owned(),
            boxes: boxes.into_owned(),
            scores: scores.into_owned(),
        })
    }
}
