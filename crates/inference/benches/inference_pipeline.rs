use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use flatbuffers::FlatBufferBuilder;
use inference::processing::{post::PostProcessor, pre::preprocess_frame};
use ndarray::{Array, IxDyn};
use schema::ColorFormat;

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

        group.bench_with_input(
            BenchmarkId::new("bgr_letterbox", format!("{}x{}", width, height)),
            &frame,
            |b, frame| {
                b.iter(|| {
                    preprocess_frame(
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
                b.iter(|| {
                    post_processor
                        .parse_detections(
                            black_box(&labels.view()),
                            black_box(&boxes.view()),
                            black_box(&scores.view()),
                            black_box(1920),
                            black_box(1080),
                            black_box(1.0),
                            black_box(0.0),
                            black_box(0.0),
                        )
                        .unwrap()
                });
            },
        );
    }

    group.finish();
}

fn benchmark_bgr_conversion(c: &mut Criterion) {
    let mut group = c.benchmark_group("bgr_conversion");

    let frame_data = create_test_frame(1920, 1080, ColorFormat::BGR);
    let frame = flatbuffers::root::<schema::Frame>(&frame_data).unwrap();

    group.bench_function("bgr_to_rgb_1920x1080", |b| {
        b.iter(|| {
            preprocess_frame(
                black_box(frame.pixels().unwrap()),
                black_box(1920),
                black_box(1080),
                black_box(ColorFormat::BGR),
            )
            .unwrap()
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_preprocessing,
    benchmark_postprocessing,
    benchmark_bgr_conversion
);
criterion_main!(benches);
