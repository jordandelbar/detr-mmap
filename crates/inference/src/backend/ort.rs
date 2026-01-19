use super::{InferenceBackend, InferenceOutput};
use crate::config::ExecutionProvider;
use ndarray::{Array, IxDyn};
use ort::{
    session::{Session, builder::GraphOptimizationLevel},
    value::TensorRef,
};

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

        tracing::info!("RF-DETR model loaded from {}", path);
        Ok(Self { session })
    }
}

impl InferenceBackend for OrtBackend {
    fn load_model(path: &str) -> anyhow::Result<Self> {
        Self::load_model_with_provider(path, ExecutionProvider::default())
    }

    fn infer(&mut self, images: &Array<f32, IxDyn>) -> anyhow::Result<InferenceOutput> {
        // RF-DETR: input -> dets, labels (logits)
        let outputs = self.session.run(ort::inputs![
            "input" => TensorRef::from_array_view(images.view())?
        ])?;

        let dets = outputs["dets"].try_extract_array()?;
        let logits = outputs["labels"].try_extract_array()?;

        Ok(InferenceOutput {
            dets: dets.into_owned(),
            logits: logits.into_owned(),
        })
    }
}
