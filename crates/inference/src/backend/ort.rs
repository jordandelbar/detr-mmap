use super::{InferenceBackend, InferenceOutput};
use ndarray::{Array, IxDyn};
use ort::{
    session::{Session, builder::GraphOptimizationLevel},
    value::TensorRef,
};

#[derive(Debug, Clone, Copy)]
pub enum ExecutionProvider {
    Cpu,
    Cuda,
}

pub struct OrtBackend {
    session: Session,
}

impl OrtBackend {
    /// Load model with specified execution provider
    pub fn load_model_with_provider(
        path: &str,
        provider: ExecutionProvider,
    ) -> anyhow::Result<Self> {
        // Initialize ORT environment (idempotent)
        let _ = ort::init().commit();

        let mut builder = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?;

        match provider {
            ExecutionProvider::Cuda => {
                tracing::info!("Initializing ONNX Runtime with CUDA execution provider");
                builder = builder.with_execution_providers([
                    ort::execution_providers::CUDAExecutionProvider::default()
                        .with_device_id(0)
                        .build()
                        .error_on_failure(),
                ])?;
            }
            ExecutionProvider::Cpu => {
                tracing::info!("Initializing ONNX Runtime with CPU execution provider");
            }
        }

        let session = builder.commit_from_file(path)?;

        tracing::info!("Model loaded from {}", path);
        Ok(Self { session })
    }
}

impl InferenceBackend for OrtBackend {
    fn load_model(path: &str) -> anyhow::Result<Self> {
        Self::load_model_with_provider(path, ExecutionProvider::Cuda)
    }

    fn infer(
        &mut self,
        images: &Array<f32, IxDyn>,
        orig_sizes: &Array<i64, IxDyn>,
    ) -> anyhow::Result<InferenceOutput> {
        let outputs = self.session.run(ort::inputs![
            "images" => TensorRef::from_array_view(images.view())?,
            "orig_target_sizes" => TensorRef::from_array_view(orig_sizes.view())?
        ])?;

        let labels = outputs["labels"].try_extract_array()?;
        let boxes = outputs["boxes"].try_extract_array()?;
        let scores = outputs["scores"].try_extract_array()?;

        Ok(InferenceOutput {
            labels: labels.into_owned(),
            boxes: boxes.into_owned(),
            scores: scores.into_owned(),
        })
    }
}
