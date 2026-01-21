use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use flatbuffers::FlatBufferBuilder;
use inference::{
    backend::{InferenceBackend, InferenceOutput},
    processing::{post::PostProcessor, pre::PreProcessor},
};
use ndarray::{Array, IxDyn};
use schema::ColorFormat;
use std::path::Path;

#[cfg(feature = "ort-backend")]
use inference::backend::ort::OrtBackend;

#[cfg(feature = "trt-backend")]
use inference::backend::trt::TrtBackend;

/// Helper function to create a FlatBuffers Frame for benchmarking
fn create_test_frame(width: u32, height: u32, format: ColorFormat) -> Vec<u8> {
    let pixel_count = (width * height * 3) as usize;
    let pixels = vec![128u8; pixel_count]; // Mid-gray image

    let mut builder = FlatBufferBuilder::new();
    let pixel_vector = builder.create_vector(&pixels);

    let frame = schema::Frame::create(
        &mut builder,
        &schema::FrameArgs {
            frame_number: 1,
            timestamp_ns: 0,
            camera_id: 0,
            width,
            height,
            channels: 3,
            format,
            pixels: Some(pixel_vector),
            trace_id: None,
            span_id: None,
            trace_flags: 0,
        },
    );

    builder.finish(frame, None);
    builder.finished_data().to_vec()
}

/// Create mock RF-DETR output with N high-confidence detections
fn create_mock_rfdetr_output(
    num_queries: usize,
    num_classes: usize,
    num_detections: usize,
) -> (ndarray::ArrayD<f32>, ndarray::ArrayD<f32>) {
    // dets: [1, num_queries, 4] - boxes in cxcywh format (normalized 0-1)
    let mut dets_data = vec![0.0f32; num_queries * 4];
    // logits: [1, num_queries, num_classes] - class logits
    let mut logits_data = vec![-10.0f32; num_queries * num_classes]; // Low confidence by default

    for i in 0..num_detections.min(num_queries) {
        // Set box coordinates (normalized cxcywh)
        dets_data[i * 4] = 0.3; // cx
        dets_data[i * 4 + 1] = 0.3; // cy
        dets_data[i * 4 + 2] = 0.2; // w
        dets_data[i * 4 + 3] = 0.2; // h

        // Set high logit for one class (will result in high confidence after sigmoid)
        let class_id = i % num_classes;
        logits_data[i * num_classes + class_id] = 5.0; // sigmoid(5) ≈ 0.99
    }

    let dets = Array::from_shape_vec(IxDyn(&[1, num_queries, 4]), dets_data).unwrap();
    let logits = Array::from_shape_vec(IxDyn(&[1, num_queries, num_classes]), logits_data).unwrap();

    (dets, logits)
}

fn benchmark_preprocessing(c: &mut Criterion) {
    let mut group = c.benchmark_group("preprocessing");

    // Test different resolutions
    let resolutions = [(640, 480), (1280, 720), (1920, 1080)];

    for (width, height) in resolutions.iter() {
        let frame_data = create_test_frame(*width, *height, ColorFormat::BGR);
        let frame = flatbuffers::root::<schema::Frame>(&frame_data).unwrap();
        let mut preprocessor = PreProcessor::default();

        group.bench_with_input(
            BenchmarkId::new("bgr_letterbox", format!("{}x{}", width, height)),
            &frame,
            |b, frame| {
                b.iter(|| {
                    preprocessor
                        .preprocess_frame(
                            black_box(frame.pixels().unwrap()),
                            black_box(frame.width()),
                            black_box(frame.height()),
                            black_box(frame.format()),
                        )
                        .unwrap()
                });
            },
        );
    }

    group.finish();
}

fn benchmark_postprocessing(c: &mut Criterion) {
    let mut group = c.benchmark_group("postprocessing");
    let post_processor = PostProcessor::new(0.5);

    let detection_counts = [0, 5, 20, 50];

    for num_detections in detection_counts.iter() {
        let (dets, logits) = create_mock_rfdetr_output(300, 91, *num_detections);

        group.bench_with_input(
            BenchmarkId::new("parse_detections", num_detections),
            &(dets, logits),
            |b, (dets, logits)| {
                let transform = inference::processing::post::TransformParams {
                    orig_width: 1920,
                    orig_height: 1080,
                    input_width: 512,
                    input_height: 512,
                    scale: 1.0,
                    offset_x: 0.0,
                    offset_y: 0.0,
                };
                b.iter(|| {
                    post_processor
                        .parse_detections(
                            black_box(&dets.view()),
                            black_box(&logits.view()),
                            black_box(&transform),
                        )
                        .unwrap()
                });
            },
        );
    }

    group.finish();
}

fn benchmark_bgr_conversion(c: &mut Criterion) {
    let mut group = c.benchmark_group("format_conversion");

    // BGR path (expensive - pixel-by-pixel conversion)
    let bgr_frame_data = create_test_frame(1920, 1080, ColorFormat::BGR);
    let bgr_frame = flatbuffers::root::<schema::Frame>(&bgr_frame_data).unwrap();
    let mut preprocessor = PreProcessor::default();

    group.bench_function("bgr_to_rgb_1920x1080", |b| {
        b.iter(|| {
            preprocessor
                .preprocess_frame(
                    black_box(bgr_frame.pixels().unwrap()),
                    black_box(1920),
                    black_box(1080),
                    black_box(ColorFormat::BGR),
                )
                .unwrap()
        });
    });

    // RGB path (production path - frames arrive as RGB from capture)
    let rgb_frame_data = create_test_frame(1920, 1080, ColorFormat::RGB);
    let rgb_frame = flatbuffers::root::<schema::Frame>(&rgb_frame_data).unwrap();

    group.bench_function("rgb_passthrough_1920x1080", |b| {
        b.iter(|| {
            preprocessor
                .preprocess_frame(
                    black_box(rgb_frame.pixels().unwrap()),
                    black_box(1920),
                    black_box(1080),
                    black_box(ColorFormat::RGB),
                )
                .unwrap()
        });
    });

    group.finish();
}

#[cfg(any(feature = "ort-backend", feature = "trt-backend"))]
fn benchmark_inference(c: &mut Criterion) {
    let mut group = c.benchmark_group("inference");

    // Create preprocessed input (512x512) - matches RF-DETR input
    let preprocessed = Array::zeros(IxDyn(&[1, 3, 512, 512]));

    #[cfg(feature = "ort-backend")]
    {
        let onnx_model_path = "../../models/rfdetr_S/rfdetr.onnx";

        if Path::new(onnx_model_path).exists() {
            // Benchmark CPU execution provider
            if let Ok(mut cpu_backend) = OrtBackend::load_model_with_provider(
                onnx_model_path,
                inference::config::ExecutionProvider::Cpu,
            ) {
                group.bench_function("ort_cpu", |b| {
                    b.iter(|| cpu_backend.infer(black_box(&preprocessed)).unwrap());
                });
            } else {
                eprintln!("Failed to load ONNX model with CPU provider");
            }

            // Benchmark CUDA execution provider
            if let Ok(mut cuda_backend) = OrtBackend::load_model_with_provider(
                onnx_model_path,
                inference::config::ExecutionProvider::Cuda,
            ) {
                group.bench_function("ort_cuda", |b| {
                    b.iter(|| cuda_backend.infer(black_box(&preprocessed)).unwrap());
                });
            } else {
                eprintln!("Failed to load ONNX model with CUDA provider");
            }
        } else {
            eprintln!(
                "Skipping ONNX inference benchmark: model not found at {}",
                onnx_model_path
            );
        }
    }

    #[cfg(feature = "trt-backend")]
    {
        let trt_model_path = "../../models/rfdetr_S/rfdetr.engine";

        if Path::new(trt_model_path).exists() {
            if let Ok(mut trt_backend) = TrtBackend::load_model(trt_model_path) {
                group.bench_function("trt", |b| {
                    b.iter(|| trt_backend.infer(black_box(&preprocessed)).unwrap());
                });
            } else {
                eprintln!("Failed to load TensorRT model");
            }
        } else {
            eprintln!(
                "Skipping TensorRT inference benchmark: model not found at {}",
                trt_model_path
            );
        }
    }

    group.finish();
}

/// End-to-end pipeline benchmark: preprocessing -> inference -> postprocessing
#[cfg(any(feature = "ort-backend", feature = "trt-backend"))]
fn benchmark_full_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_pipeline");

    // Create test frame
    let frame_data = create_test_frame(1920, 1080, ColorFormat::RGB);
    let frame = flatbuffers::root::<schema::Frame>(&frame_data).unwrap();

    let mut preprocessor = PreProcessor::new((512, 512));
    let post_processor = PostProcessor::new(0.5);

    #[cfg(feature = "ort-backend")]
    {
        let onnx_model_path = "../../models/rfdetr_S/rfdetr.onnx";

        if Path::new(onnx_model_path).exists() {
            // CPU pipeline
            if let Ok(mut cpu_backend) = OrtBackend::load_model_with_provider(
                onnx_model_path,
                inference::config::ExecutionProvider::Cpu,
            ) {
                group.bench_function("ort_cpu", |b| {
                    b.iter(|| {
                        let (preprocessed, scale, offset_x, offset_y) = preprocessor
                            .preprocess_frame(
                                black_box(frame.pixels().unwrap()),
                                black_box(frame.width()),
                                black_box(frame.height()),
                                black_box(frame.format()),
                            )
                            .unwrap();

                        let transform = inference::processing::post::TransformParams {
                            orig_width: 1920,
                            orig_height: 1080,
                            input_width: 512,
                            input_height: 512,
                            scale,
                            offset_x,
                            offset_y,
                        };

                        let InferenceOutput { dets, logits } =
                            cpu_backend.infer(black_box(&preprocessed)).unwrap();

                        post_processor
                            .parse_detections(
                                black_box(&dets.view()),
                                black_box(&logits.view()),
                                black_box(&transform),
                            )
                            .unwrap()
                    });
                });
            } else {
                eprintln!("Failed to load ONNX model with CPU provider");
            }

            // CUDA pipeline
            if let Ok(mut cuda_backend) = OrtBackend::load_model_with_provider(
                onnx_model_path,
                inference::config::ExecutionProvider::Cuda,
            ) {
                group.bench_function("ort_cuda", |b| {
                    b.iter(|| {
                        let (preprocessed, scale, offset_x, offset_y) = preprocessor
                            .preprocess_frame(
                                black_box(frame.pixels().unwrap()),
                                black_box(frame.width()),
                                black_box(frame.height()),
                                black_box(frame.format()),
                            )
                            .unwrap();

                        let transform = inference::processing::post::TransformParams {
                            orig_width: 1920,
                            orig_height: 1080,
                            input_width: 512,
                            input_height: 512,
                            scale,
                            offset_x,
                            offset_y,
                        };

                        let InferenceOutput { dets, logits } =
                            cuda_backend.infer(black_box(&preprocessed)).unwrap();

                        post_processor
                            .parse_detections(
                                black_box(&dets.view()),
                                black_box(&logits.view()),
                                black_box(&transform),
                            )
                            .unwrap()
                    });
                });
            } else {
                eprintln!("Failed to load ONNX model with CUDA provider");
            }
        } else {
            eprintln!(
                "Skipping ONNX full pipeline benchmark: model not found at {}",
                onnx_model_path
            );
        }
    }

    #[cfg(feature = "trt-backend")]
    {
        let trt_model_path = "../../models/rfdetr_S/rfdetr.engine";

        if Path::new(trt_model_path).exists() {
            if let Ok(mut trt_backend) = TrtBackend::load_model(trt_model_path) {
                group.bench_function("trt", |b| {
                    b.iter(|| {
                        let (preprocessed, scale, offset_x, offset_y) = preprocessor
                            .preprocess_frame(
                                black_box(frame.pixels().unwrap()),
                                black_box(frame.width()),
                                black_box(frame.height()),
                                black_box(frame.format()),
                            )
                            .unwrap();

                        let transform = inference::processing::post::TransformParams {
                            orig_width: 1920,
                            orig_height: 1080,
                            input_width: 512,
                            input_height: 512,
                            scale,
                            offset_x,
                            offset_y,
                        };

                        let InferenceOutput { dets, logits } =
                            trt_backend.infer(black_box(&preprocessed)).unwrap();

                        post_processor
                            .parse_detections(
                                black_box(&dets.view()),
                                black_box(&logits.view()),
                                black_box(&transform),
                            )
                            .unwrap()
                    });
                });
            } else {
                eprintln!("Failed to load TensorRT model");
            }
        } else {
            eprintln!(
                "Skipping TensorRT full pipeline benchmark: model not found at {}",
                trt_model_path
            );
        }
    }

    group.finish();
}

// Conditionally include backend benchmarks based on features
#[cfg(any(feature = "ort-backend", feature = "trt-backend"))]
criterion_group!(
    benches,
    benchmark_preprocessing,
    benchmark_postprocessing,
    benchmark_bgr_conversion,
    benchmark_inference,
    benchmark_full_pipeline
);

criterion_main!(benches);
