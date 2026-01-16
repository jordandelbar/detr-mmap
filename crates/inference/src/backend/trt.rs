use super::{InferenceBackend, InferenceOutput};
use ndarray::{Array, IxDyn};

use std::ffi::CString;

#[cxx::bridge]
mod ffi {
    unsafe extern "C++" {
        include!("rfdetr_backend.hpp");
        include!("logging.hpp");

        #[namespace = "bridge"]
        type RFDetrBackend;

        #[namespace = "bridge"]
        fn new_rfdetr_backend() -> UniquePtr<RFDetrBackend>;

        #[namespace = "bridge"]
        fn init_logger();

        // Maps to RFDetrBackend::load_engine
        #[namespace = "bridge"]
        unsafe fn load_engine(self: Pin<&mut RFDetrBackend>, path: *const c_char) -> bool;

        // Maps to RFDetrBackend::infer_raw
        #[namespace = "bridge"]
        unsafe fn infer_raw(
            self: Pin<&mut RFDetrBackend>,
            images: *const f32,
            out_dets: *mut f32,
            out_logits: *mut f32,
        ) -> bool;

        #[namespace = "bridge"]
        fn get_num_queries(self: &RFDetrBackend) -> i32;

        #[namespace = "bridge"]
        fn get_num_classes(self: &RFDetrBackend) -> i32;
    }
}

pub struct TrtBackend {
    inner: cxx::UniquePtr<ffi::RFDetrBackend>,
    num_queries: usize,
    num_classes: usize,
}

impl InferenceBackend for TrtBackend {
    fn load_model(path: &str) -> anyhow::Result<Self> {
        ffi::init_logger();
        let mut inner = ffi::new_rfdetr_backend();

        if inner.is_null() {
            return Err(anyhow::anyhow!(
                "Failed to create RF-DETR TensorRT backend instance"
            ));
        }

        let c_path = CString::new(path)?;

        if !unsafe { inner.pin_mut().load_engine(c_path.as_ptr()) } {
            return Err(anyhow::anyhow!(
                "Failed to load RF-DETR TensorRT engine from {}",
                path
            ));
        }

        let num_queries = inner.get_num_queries() as usize;
        let num_classes = inner.get_num_classes() as usize;

        Ok(Self {
            inner,
            num_queries,
            num_classes,
        })
    }

    fn infer(&mut self, images: &Array<f32, IxDyn>) -> anyhow::Result<InferenceOutput> {
        // Prepare output buffers
        // Dets: [1, num_queries, 4]
        let mut dets = Array::<f32, IxDyn>::zeros(IxDyn(&[1, self.num_queries, 4]));
        // Logits: [1, num_queries, num_classes]
        let mut logits =
            Array::<f32, IxDyn>::zeros(IxDyn(&[1, self.num_queries, self.num_classes]));

        let success = unsafe {
            self.inner
                .pin_mut()
                .infer_raw(images.as_ptr(), dets.as_mut_ptr(), logits.as_mut_ptr())
        };

        if !success {
            return Err(anyhow::anyhow!("RF-DETR TensorRT inference failed"));
        }

        Ok(InferenceOutput { dets, logits })
    }
}
