use crate::{
    backend::{InferenceBackend, InferenceOutput},
    config::InferenceConfig,
    processing::{
        post::{PostProcessor, TransformParams},
        pre::PreProcessor,
    },
};
use bridge::{BridgeSemaphore, DetectionWriter, FrameReader, SemaphoreType, TraceContextBytes};
use common::wait_for_resource;
use opentelemetry::{
    global,
    metrics::{Counter, Histogram},
    trace::TraceContextExt,
};
use std::thread;
use std::time::{Duration, Instant};
use tracing_opentelemetry::OpenTelemetrySpanExt;

pub struct InferenceService<B: InferenceBackend> {
    backend: B,
    config: InferenceConfig,
    postprocessor: PostProcessor,
    preprocessor: PreProcessor,
}

fn init_metrics(
    meter_name: &'static str,
) -> (Histogram<f64>, Counter<u64>, Counter<u64>, Counter<u64>) {
    let meter = global::meter(meter_name);
    let latency_buckets = [
        0.001, 0.002, 0.005, 0.007, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.075, 0.1, 0.15,
        0.2, 0.5,
    ];
    let duration_histogram: Histogram<f64> = meter
        .f64_histogram("inference_duration_seconds")
        .with_description("Time to process a single frame (preprocess + infer + postprocess)")
        .with_unit("s")
        .with_boundaries(latency_buckets.to_vec())
        .build();
    let frames_counter: Counter<u64> = meter
        .u64_counter("inference_frames_total")
        .with_description("Total frames processed")
        .build();
    let skipped_counter: Counter<u64> = meter
        .u64_counter("inference_frames_skipped_total")
        .with_description("Total frames skipped (processing too slow)")
        .build();
    let detections_counter: Counter<u64> = meter
        .u64_counter("inference_detections_total")
        .with_description("Total detections produced")
        .build();

    (
        duration_histogram,
        frames_counter,
        skipped_counter,
        detections_counter,
    )
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

        let (duration_histogram, frames_counter, skipped_counter, detections_counter) =
            init_metrics("inference");

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
                        skipped_counter.add(skipped as u64, &[]);
                        tracing::trace!(skipped, "Skipped frames to process latest");
                    }
                }
                Err(e) => {
                    tracing::warn!(error = %e, "Failed to drain semaphore");
                }
            }

            let start = Instant::now();
            match self.process_frame(&frame_reader, &mut detection_writer) {
                Ok(detections) => {
                    let elapsed = start.elapsed().as_secs_f64();
                    duration_histogram.record(elapsed, &[]);
                    frames_counter.add(1, &[]);
                    detections_counter.add(detections as u64, &[]);

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
        let (frame, trace_ctx) = frame_reader
            .get_frame_with_context()?
            .ok_or_else(|| anyhow::anyhow!("No frame available"))?;

        let frame_num = frame.frame_number();
        let timestamp_ns = frame.timestamp_ns();
        let camera_id = frame.camera_id();
        let width = frame.width();
        let height = frame.height();

        // Create span linked to capture trace
        let span = tracing::info_span!(
            "inference_process_frame",
            frame_number = frame_num,
            camera_id = camera_id
        );

        // Link to parent trace from capture if available
        if let Some(ref ctx) = trace_ctx {
            let parent_ctx = trace_ctx_to_otel_context(ctx);
            let _ = span.set_parent(parent_ctx);
        }

        let _enter = span.enter();

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

        let InferenceOutput { dets, logits } = {
            let _infer_span = tracing::info_span!("model_inference").entered();
            self.backend.infer(&preprocessed)?
        };

        let transform = TransformParams {
            orig_width: width,
            orig_height: height,
            input_width: self.config.input_size.0,
            input_height: self.config.input_size.1,
            scale,
            offset_x,
            offset_y,
        };

        let detections =
            self.postprocessor
                .parse_detections(&dets.view(), &logits.view(), &transform)?;

        // Write detections with trace context for downstream propagation
        detection_writer.write_with_trace_context(
            frame_num,
            timestamp_ns,
            camera_id,
            &detections,
            trace_ctx.as_ref(),
        )?;

        Ok(detections.len())
    }
}

/// Convert TraceContextBytes to an OpenTelemetry Context for span linking.
fn trace_ctx_to_otel_context(ctx: &TraceContextBytes) -> opentelemetry::Context {
    use opentelemetry::trace::{SpanContext, SpanId, TraceFlags, TraceId, TraceState};

    let span_context = SpanContext::new(
        TraceId::from_bytes(ctx.trace_id),
        SpanId::from_bytes(ctx.span_id),
        TraceFlags::new(ctx.trace_flags),
        true, // remote = true since this came from another process
        TraceState::default(),
    );

    opentelemetry::Context::new().with_remote_span_context(span_context)
}
