use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use flatbuffers::FlatBufferBuilder;
use inference::{
    backend::InferenceBackend,
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
        },
    );

    builder.finish(frame, None);
    builder.finished_data().to_vec()
}

/// Create mock RT-DETR output with N high-confidence detections
fn create_mock_rtdetr_output(
    num_queries: usize,
    num_detections: usize,
) -> (
    ndarray::ArrayD<i64>,
    ndarray::ArrayD<f32>,
    ndarray::ArrayD<f32>,
) {
    let mut label_data = vec![0i64; num_queries];
    let mut box_data = vec![0.0f32; num_queries * 4];
    let mut score_data = vec![0.01f32; num_queries];

    for i in 0..num_detections.min(num_queries) {
        label_data[i] = (i % 80) as i64;
        box_data[i * 4] = 100.0;
        box_data[i * 4 + 1] = 100.0;
        box_data[i * 4 + 2] = 200.0;
        box_data[i * 4 + 3] = 200.0;
        score_data[i] = 0.9;
    }

    let labels = Array::from_shape_vec(IxDyn(&[1, num_queries]), label_data).unwrap();
    let boxes = Array::from_shape_vec(IxDyn(&[1, num_queries, 4]), box_data).unwrap();
    let scores = Array::from_shape_vec(IxDyn(&[1, num_queries]), score_data).unwrap();

    (labels, boxes, scores)
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
        let (labels, boxes, scores) = create_mock_rtdetr_output(300, *num_detections);

        group.bench_with_input(
            BenchmarkId::new("parse_detections", num_detections),
            &(labels, boxes, scores),
            |b, (labels, boxes, scores)| {
                let transform = inference::processing::post::TransformParams {
                    orig_width: 1920,
                    orig_height: 1080,
                    scale: 1.0,
                    offset_x: 0.0,
                    offset_y: 0.0,
                };
                b.iter(|| {
                    post_processor
                        .parse_detections(
                            black_box(&labels.view()),
                            black_box(&boxes.view()),
                            black_box(&scores.view()),
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

    // RGB path (this is production path with nokhwa)
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

    // Create preprocessed input (640x640) - matches RT-DETR input
    let preprocessed = Array::zeros(IxDyn(&[1, 3, 640, 640]));

    // RT-DETR expects original target sizes
    let orig_sizes = Array::from_shape_vec((1, 2), vec![640i64, 640i64])
        .unwrap()
        .into_dyn();

    #[cfg(feature = "ort-backend")]
    {
        let onnx_model_path = "../../models/model.onnx";

        if Path::new(onnx_model_path).exists() {
            // Benchmark CPU execution provider
            if let Ok(mut cpu_backend) = OrtBackend::load_model_with_provider(
                onnx_model_path,
                inference::backend::ort::ExecutionProvider::Cpu,
            ) {
                group.bench_function("ort_cpu", |b| {
                    b.iter(|| {
                        cpu_backend
                            .infer(black_box(&preprocessed), black_box(&orig_sizes))
                            .unwrap()
                    });
                });
            } else {
                eprintln!("Failed to load ONNX model with CPU provider");
            }

            // Benchmark CUDA execution provider
            if let Ok(mut cuda_backend) = OrtBackend::load_model_with_provider(
                onnx_model_path,
                inference::backend::ort::ExecutionProvider::Cuda,
            ) {
                group.bench_function("ort_cuda", |b| {
                    b.iter(|| {
                        cuda_backend
                            .infer(black_box(&preprocessed), black_box(&orig_sizes))
                            .unwrap()
                    });
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
        let trt_model_path = "../../models/model_fp16.engine";

        if Path::new(trt_model_path).exists() {
            if let Ok(mut trt_backend) = TrtBackend::load_model(trt_model_path) {
                group.bench_function("trt", |b| {
                    b.iter(|| {
                        trt_backend
                            .infer(black_box(&preprocessed), black_box(&orig_sizes))
                            .unwrap()
                    });
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

    let mut preprocessor = PreProcessor::new((640, 640));
    let post_processor = PostProcessor::new(0.5);

    #[cfg(feature = "ort-backend")]
    {
        let onnx_model_path = "../../models/model.onnx";

        if Path::new(onnx_model_path).exists() {
            // CPU pipeline
            if let Ok(mut cpu_backend) = OrtBackend::load_model_with_provider(
                onnx_model_path,
                inference::backend::ort::ExecutionProvider::Cpu,
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
                            scale,
                            offset_x,
                            offset_y,
                        };

                        let orig_sizes = Array::from_shape_vec((1, 2), vec![640i64, 640i64])
                            .unwrap()
                            .into_dyn();
                        let outputs = cpu_backend
                            .infer(black_box(&preprocessed), black_box(&orig_sizes))
                            .unwrap();

                        post_processor
                            .parse_detections(
                                black_box(&outputs.labels.view()),
                                black_box(&outputs.boxes.view()),
                                black_box(&outputs.scores.view()),
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
                inference::backend::ort::ExecutionProvider::Cuda,
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
                            scale,
                            offset_x,
                            offset_y,
                        };

                        let orig_sizes = Array::from_shape_vec((1, 2), vec![640i64, 640i64])
                            .unwrap()
                            .into_dyn();
                        let outputs = cuda_backend
                            .infer(black_box(&preprocessed), black_box(&orig_sizes))
                            .unwrap();

                        post_processor
                            .parse_detections(
                                black_box(&outputs.labels.view()),
                                black_box(&outputs.boxes.view()),
                                black_box(&outputs.scores.view()),
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
        let trt_model_path = "../../models/model_fp16.engine";

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
                            scale,
                            offset_x,
                            offset_y,
                        };

                        let orig_sizes = Array::from_shape_vec((1, 2), vec![640i64, 640i64])
                            .unwrap()
                            .into_dyn();
                        let outputs = trt_backend
                            .infer(black_box(&preprocessed), black_box(&orig_sizes))
                            .unwrap();

                        post_processor
                            .parse_detections(
                                black_box(&outputs.labels.view()),
                                black_box(&outputs.boxes.view()),
                                black_box(&outputs.scores.view()),
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

#[cfg(not(any(feature = "ort-backend", feature = "trt-backend")))]
criterion_group!(
    benches,
    benchmark_preprocessing,
    benchmark_postprocessing,
    benchmark_bgr_conversion
);

criterion_main!(benches);
