use crate::{
    backend::InferenceBackend,
    config::InferenceConfig,
    postprocessing::{build_detection_flatbuffer, parse_detections},
    preprocessing::preprocess_frame,
};
use bridge::{FrameWriter, MmapReader};
use ndarray::Array;
use std::thread;
use std::time::Duration;

pub struct InferenceService<B: InferenceBackend> {
    backend: B,
    config: InferenceConfig,
}

impl<B: InferenceBackend> InferenceService<B> {
    pub fn new(backend: B, config: InferenceConfig) -> Self {
        Self { backend, config }
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
                    thread::sleep(Duration::from_millis(100));
                }
            }
        };

        tracing::info!(
            detection_buffer = %self.config.detection_mmap_path,
            size_mb = self.config.detection_mmap_size / 1024 / 1024,
            "Creating detection buffer"
        );

        let mut detection_writer =
            FrameWriter::new(&self.config.detection_mmap_path, self.config.detection_mmap_size)?;

        tracing::info!(
            poll_interval_ms = self.config.poll_interval_ms,
            "Starting inference loop"
        );

        let mut total_detections = 0usize;
        let mut frames_processed = 0u64;

        loop {
            if !frame_reader.has_new_data() {
                thread::sleep(Duration::from_millis(self.config.poll_interval_ms));
                continue;
            }

            match self.process_frame(&frame_reader, &mut detection_writer) {
                Ok(detections) => {
                    frames_processed += 1;
                    total_detections += detections;

                    tracing::debug!(
                        frames_processed,
                        total_detections,
                        detections,
                        "Frame processed"
                    );
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
        detection_writer: &mut FrameWriter,
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

        let (preprocessed, scale, offset_x, offset_y) =
            preprocess_frame(pixels, width, height, format)?;

        let orig_sizes = Array::from_shape_vec(
            (1, 2),
            vec![self.config.input_size.1 as i64, self.config.input_size.0 as i64],
        )?
        .into_dyn();

        tracing::trace!(frame_num, "Running inference");
        let output = self.backend.infer(&preprocessed, &orig_sizes)?;

        let detections = parse_detections(
            &output.labels.view(),
            &output.boxes.view(),
            &output.scores.view(),
            width,
            height,
            scale,
            offset_x,
            offset_y,
        )?;

        let detection_buffer =
            build_detection_flatbuffer(frame_num, timestamp_ns, camera_id, &detections)?;

        detection_writer.write(&detection_buffer)?;

        Ok(detections.len())
    }
}
