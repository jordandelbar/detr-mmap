use super::{InferenceBackend, InferenceOutput};
use ndarray::{Array, IxDyn};

use std::ffi::CString;

#[cxx::bridge]
mod ffi {
    unsafe extern "C++" {
        include!("tensorrt_backend.hpp");

        #[namespace = "bridge"]
        type TensorRTBackend;

        #[namespace = "bridge"]
        fn new_tensorrt_backend() -> UniquePtr<TensorRTBackend>;

        // Maps to TensorRTBackend::load_engine
        #[namespace = "bridge"]
        unsafe fn load_engine(self: Pin<&mut TensorRTBackend>, path: *const c_char) -> bool;

        // Maps to TensorRTBackend::infer_raw
        #[namespace = "bridge"]
        unsafe fn infer_raw(
            self: Pin<&mut TensorRTBackend>,
            images: *const f32,
            orig_sizes: *const i64,
            out_labels: *mut i64,
            out_boxes: *mut f32,
            out_scores: *mut f32,
        ) -> bool;

        #[namespace = "bridge"]
        fn get_num_detections(self: &TensorRTBackend) -> i32;
    }
}

pub struct TrtBackend {
    inner: cxx::UniquePtr<ffi::TensorRTBackend>,
    num_detections: usize,
}

impl InferenceBackend for TrtBackend {
    fn load_model(path: &str) -> anyhow::Result<Self> {
        let mut inner = ffi::new_tensorrt_backend();

        if inner.is_null() {
            return Err(anyhow::anyhow!(
                "Failed to create TensorRT backend instance"
            ));
        }

        let c_path = CString::new(path)?;

        if !unsafe { inner.pin_mut().load_engine(c_path.as_ptr()) } {
            return Err(anyhow::anyhow!(
                "Failed to load TensorRT engine from {}",
                path
            ));
        }

        let num_detections = inner.get_num_detections() as usize;

        Ok(Self {
            inner,
            num_detections,
        })
    }

    fn infer(
        &mut self,
        images: &Array<f32, IxDyn>,
        orig_sizes: &Array<i64, IxDyn>,
    ) -> anyhow::Result<InferenceOutput> {
        // Prepare output buffers
        // Labels: [1, num_detections]
        let mut labels = Array::<i64, IxDyn>::zeros(IxDyn(&[1, self.num_detections]));
        // Boxes: [1, num_detections, 4]
        let mut boxes = Array::<f32, IxDyn>::zeros(IxDyn(&[1, self.num_detections, 4]));
        // Scores: [1, num_detections]
        let mut scores = Array::<f32, IxDyn>::zeros(IxDyn(&[1, self.num_detections]));

        let success = unsafe {
            self.inner.pin_mut().infer_raw(
                images.as_ptr(),
                orig_sizes.as_ptr(),
                labels.as_mut_ptr(),
                boxes.as_mut_ptr(),
                scores.as_mut_ptr(),
            )
        };

        if !success {
            return Err(anyhow::anyhow!("TensorRT inference failed"));
        }

        Ok(InferenceOutput {
            labels,
            boxes,
            scores,
        })
    }
}
