use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use flatbuffers::FlatBufferBuilder;
use preprocess::CpuPreProcessor;

#[cfg(feature = "cuda")]
use preprocess::GpuPreProcessor;

/// Helper function to create a FlatBuffers Frame for benchmarking
fn create_test_frame(width: u32, height: u32) -> Vec<u8> {
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
            pixels: Some(pixel_vector),
            trace: None,
        },
    );

    builder.finish(frame, None);
    builder.finished_data().to_vec()
}

/// Create raw pixel buffer for benchmarking (gradient pattern)
fn create_test_pixels(width: u32, height: u32) -> Vec<u8> {
    let mut pixels = vec![0u8; (width * height * 3) as usize];
    for y in 0..height {
        for x in 0..width {
            let idx = ((y * width + x) * 3) as usize;
            pixels[idx] = (x % 256) as u8; // R
            pixels[idx + 1] = (y % 256) as u8; // G
            pixels[idx + 2] = ((x + y) % 256) as u8; // B
        }
    }
    pixels
}

fn benchmark_cpu_preprocess(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_preprocess");

    let resolutions = [(640, 480), (1280, 720), (1920, 1080), (3840, 2160)];
    let input_size = (512, 512);

    let mut preprocessor = CpuPreProcessor::new(input_size);

    for (width, height) in resolutions.iter() {
        let pixels = create_test_pixels(*width, *height);

        group.bench_with_input(
            BenchmarkId::new("letterbox", format!("{}x{}", width, height)),
            &pixels,
            |b, pixels| {
                b.iter(|| {
                    preprocessor
                        .preprocess_from_u8_slice(
                            black_box(pixels),
                            black_box(*width),
                            black_box(*height),
                        )
                        .unwrap()
                });
            },
        );
    }

    group.finish();
}

fn benchmark_cpu_preprocess_frame(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_preprocess_frame");

    let resolutions = [(640, 480), (1280, 720), (1920, 1080)];

    for (width, height) in resolutions.iter() {
        let frame_data = create_test_frame(*width, *height);
        let frame = flatbuffers::root::<schema::Frame>(&frame_data).unwrap();
        let mut preprocessor = CpuPreProcessor::default();

        group.bench_with_input(
            BenchmarkId::new("letterbox", format!("{}x{}", width, height)),
            &frame,
            |b, frame| {
                b.iter(|| {
                    preprocessor
                        .preprocess_frame(
                            black_box(frame.pixels().unwrap()),
                            black_box(frame.width()),
                            black_box(frame.height()),
                        )
                        .unwrap()
                });
            },
        );
    }

    group.finish();
}

#[cfg(feature = "cuda")]
fn benchmark_gpu_vs_cpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("preprocess_comparison");

    let width = 1920u32;
    let height = 1080u32;
    let input_size = (512, 512);
    let pixels = create_test_pixels(width, height);

    // CPU benchmark
    let mut cpu_preprocessor = CpuPreProcessor::new(input_size);
    group.bench_function("cpu_1080p", |b| {
        b.iter(|| {
            cpu_preprocessor
                .preprocess_from_u8_slice(black_box(&pixels), black_box(width), black_box(height))
                .unwrap()
        });
    });

    // GPU benchmark (kernel only, data pre-uploaded)
    if let Ok(mut gpu_preprocessor) = GpuPreProcessor::new(input_size, (width, height)) {
        gpu_preprocessor
            .upload_to_device(&pixels, width, height)
            .unwrap();
        group.bench_function("gpu_1080p", |b| {
            b.iter(|| {
                gpu_preprocessor
                    .run_kernel(black_box(width), black_box(height))
                    .unwrap()
            });
        });
    } else {
        eprintln!("GPU not available for comparison benchmark");
    }

    group.finish();
}

#[cfg(feature = "cuda")]
fn benchmark_gpu_preprocess(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_preprocess");

    let resolutions = [(640, 480), (1280, 720), (1920, 1080), (3840, 2160)];
    let input_size = (512, 512);
    let max_input_size = (3840, 2160);

    let gpu_result = GpuPreProcessor::new(input_size, max_input_size);
    if let Err(e) = &gpu_result {
        eprintln!("Skipping GPU kernel-only benchmark: {}", e);
        group.finish();
        return;
    }
    let mut gpu_preprocessor = gpu_result.unwrap();

    for (width, height) in resolutions.iter() {
        let pixels = create_test_pixels(*width, *height);

        // Pre-upload data to device (not timed)
        gpu_preprocessor
            .upload_to_device(&pixels, *width, *height)
            .unwrap();

        group.bench_with_input(
            BenchmarkId::new("letterbox", format!("{}x{}", width, height)),
            &(*width, *height),
            |b, &(w, h)| {
                b.iter(|| {
                    gpu_preprocessor
                        .run_kernel(black_box(w), black_box(h))
                        .unwrap()
                });
            },
        );
    }

    group.finish();
}

#[cfg(feature = "cuda")]
criterion_group!(
    benches,
    benchmark_cpu_preprocess,
    benchmark_cpu_preprocess_frame,
    benchmark_gpu_preprocess,
    benchmark_gpu_vs_cpu
);

#[cfg(not(feature = "cuda"))]
criterion_group!(
    benches,
    benchmark_cpu_preprocess,
    benchmark_cpu_preprocess_frame
);

criterion_main!(benches);
