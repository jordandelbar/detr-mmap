use crate::{
    backend::InferenceBackend,
    config::InferenceConfig,
    processing::{
        post::{PostProcessor, TransformParams},
        pre::PreProcessor,
    },
    serialization::DetectionSerializer,
};
use bridge::{FrameSemaphore, MmapReader};
use ndarray::Array;
use std::thread;
use std::time::Duration;

pub struct InferenceService<B: InferenceBackend> {
    backend: B,
    config: InferenceConfig,
    postprocessor: PostProcessor,
    preprocessor: PreProcessor,
}

impl<B: InferenceBackend> InferenceService<B> {
    pub fn new(backend: B, config: InferenceConfig) -> Self {
        let postprocessor = PostProcessor {
            confidence_threshold: config.confidence_threshold,
        };
        let preprocessor = PreProcessor {
            input_size: config.input_size,
        };
        Self {
            backend,
            config,
            postprocessor,
            preprocessor,
        }
    }

    pub fn run(mut self) -> anyhow::Result<()> {
        tracing::info!(
            model_path = %self.config.model_path,
            "Inference service starting"
        );

        tracing::info!(
            frame_buffer = %self.config.frame_mmap_path,
            "Waiting for frame buffer connection"
        );

        let mut frame_reader = loop {
            match MmapReader::new(&self.config.frame_mmap_path) {
                Ok(reader) => {
                    tracing::info!("Frame buffer connected successfully");
                    break reader;
                }
                Err(_) => {
                    thread::sleep(Duration::from_millis(self.config.poll_interval_ms));
                }
            }
        };

        tracing::info!(
            detection_buffer = %self.config.detection_mmap_path,
            size_mb = self.config.detection_mmap_size / 1024 / 1024,
            "Creating detection buffer"
        );

        let mut detection_serializer = DetectionSerializer::build(
            &self.config.detection_mmap_path,
            self.config.detection_mmap_size,
        )?;

        tracing::info!("Opening inference frame synchronization semaphore");
        let frame_semaphore = loop {
            match FrameSemaphore::open("/bridge_frame_inference") {
                Ok(sem) => {
                    tracing::info!("Inference semaphore connected successfully");
                    break sem;
                }
                Err(_) => {
                    thread::sleep(Duration::from_millis(self.config.poll_interval_ms));
                }
            }
        };

        tracing::info!("Opening controller semaphore for detection notifications");
        let controller_semaphore = loop {
            match FrameSemaphore::open(&self.config.controller_semaphore_name) {
                Ok(sem) => {
                    tracing::info!("Controller semaphore connected successfully");
                    break sem;
                }
                Err(_) => {
                    match FrameSemaphore::create(&self.config.controller_semaphore_name) {
                        Ok(sem) => {
                            tracing::info!("Controller semaphore created successfully");
                            break sem;
                        }
                        Err(_) => {
                            thread::sleep(Duration::from_millis(self.config.poll_interval_ms));
                        }
                    }
                }
            }
        };

        tracing::info!("Starting inference loop (event-driven)");

        let mut total_detections = 0usize;
        let mut frames_processed = 0u64;
        let mut frames_skipped = 0u64;

        loop {
            // Wait for frame ready signal
            if let Err(e) = frame_semaphore.wait() {
                tracing::error!(error = %e, "Semaphore wait failed");
                thread::sleep(Duration::from_millis(self.config.poll_interval_ms));
                continue;
            }

            // Drain any additional pending signals to skip to the latest frame
            match frame_semaphore.drain() {
                Ok(skipped) => {
                    if skipped > 0 {
                        frames_skipped += skipped as u64;
                        tracing::trace!(skipped, "Skipped frames to process latest");
                    }
                }
                Err(e) => {
                    tracing::warn!(error = %e, "Failed to drain semaphore");
                }
            }

            match self.process_frame(&frame_reader, &mut detection_serializer) {
                Ok(detections) => {
                    frames_processed += 1;
                    total_detections += detections;

                    if let Err(e) = controller_semaphore.post() {
                        tracing::warn!(error = %e, "Failed to signal controller");
                    }

                    if frames_processed.is_multiple_of(10) {
                        tracing::debug!(
                            frames_processed,
                            frames_skipped,
                            total_detections,
                            detections,
                            "Frame processed"
                        );
                    }
                }
                Err(e) => {
                    tracing::error!(error = %e, "Failed to process frame");
                }
            }

            frame_reader.mark_read();
        }
    }

    fn process_frame(
        &mut self,
        frame_reader: &MmapReader,
        detection_serializer: &mut DetectionSerializer,
    ) -> anyhow::Result<usize> {
        let frame = flatbuffers::root::<schema::Frame>(frame_reader.buffer())?;
        let frame_num = frame.frame_number();
        let timestamp_ns = frame.timestamp_ns();
        let camera_id = frame.camera_id();
        let width = frame.width();
        let height = frame.height();

        let pixels = frame
            .pixels()
            .ok_or_else(|| anyhow::anyhow!("No pixel data"))?;
        let format = frame.format();

        tracing::trace!(
            frame_num,
            width,
            height,
            format = ?format,
            "Preprocessing frame"
        );

        let (preprocessed, scale, offset_x, offset_y) = self
            .preprocessor
            .preprocess_frame(pixels, width, height, format)?;

        let orig_sizes = Array::from_shape_vec(
            (1, 2),
            vec![
                self.config.input_size.1 as i64,
                self.config.input_size.0 as i64,
            ],
        )?
        .into_dyn();

        tracing::trace!(frame_num, "Running inference");
        let output = self.backend.infer(&preprocessed, &orig_sizes)?;

        let transform = TransformParams {
            orig_width: width,
            orig_height: height,
            scale,
            offset_x,
            offset_y,
        };
        let detections = self.postprocessor.parse_detections(
            &output.labels.view(),
            &output.boxes.view(),
            &output.scores.view(),
            &transform,
        )?;

        detection_serializer.write(frame_num, timestamp_ns, camera_id, &detections)?;

        Ok(detections.len())
    }
}
