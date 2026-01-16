use crate::{
    backend::{InferenceBackend, InferenceOutput},
    config::InferenceConfig,
    processing::{
        post::{PostProcessor, TransformParams},
        pre::PreProcessor,
    },
};
use bridge::{BridgeSemaphore, DetectionWriter, FrameReader, SemaphoreType};
use common::wait_for_resource;
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
        let postprocessor = PostProcessor::new(config.confidence_threshold);
        let preprocessor = PreProcessor::new(config.input_size);
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

        let mut frame_reader = wait_for_resource(
            FrameReader::build,
            self.config.poll_interval_ms,
            "Frame buffer",
        );

        let mut detection_writer = DetectionWriter::build()?;

        let frame_semaphore = wait_for_resource(
            || BridgeSemaphore::open(SemaphoreType::FrameCaptureToInference),
            self.config.poll_interval_ms,
            "Inference semaphore",
        );

        let controller_semaphore = wait_for_resource(
            || BridgeSemaphore::ensure(SemaphoreType::DetectionInferenceToController),
            self.config.poll_interval_ms,
            "Controller semaphore",
        );

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

            match self.process_frame(&frame_reader, &mut detection_writer) {
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
        frame_reader: &FrameReader,
        detection_writer: &mut DetectionWriter,
    ) -> anyhow::Result<usize> {
        let frame = frame_reader
            .get_frame()?
            .ok_or_else(|| anyhow::anyhow!("No frame available"))?;
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

        tracing::trace!(frame_num, "Running inference");
        let InferenceOutput { dets, logits } = self.backend.infer(&preprocessed)?;

        let transform = TransformParams {
            orig_width: width,
            orig_height: height,
            input_width: self.config.input_size.0,
            input_height: self.config.input_size.1,
            scale,
            offset_x,
            offset_y,
        };

        let detections = self.postprocessor.parse_detections(
            &dets.view(),
            &logits.view(),
            &transform,
        )?;

        detection_writer.write(frame_num, timestamp_ns, camera_id, &detections)?;

        Ok(detections.len())
    }
}
