use super::{InferenceBackend, InferenceOutput};
use ndarray::{Array, IxDyn};
use ort::{
    session::{Session, builder::GraphOptimizationLevel},
    value::TensorRef,
};

pub struct OrtBackend {
    session: Session,
}

impl InferenceBackend for OrtBackend {
    fn load_model(path: &str) -> anyhow::Result<Self> {
        ort::init()
            .with_execution_providers([
                ort::execution_providers::TensorRTExecutionProvider::default()
                    .with_device_id(0)
                    .build(),
                ort::execution_providers::CUDAExecutionProvider::default()
                    .with_device_id(0)
                    .build(),
            ])
            .commit();

        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(path)?;

        tracing::info!("Model loaded from {}", path);
        Ok(Self { session })
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
