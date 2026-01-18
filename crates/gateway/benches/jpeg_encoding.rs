use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use gateway::polling::pixels_to_jpeg;

/// Create test pixel data with a gradient pattern (more realistic than solid color)
fn gradient_pixels(width: u32, height: u32) -> Vec<u8> {
    let size = (width * height * 3) as usize;
    let mut data = Vec::with_capacity(size);
    for y in 0..height {
        for x in 0..width {
            let r = ((x * 255) / width) as u8;
            let g = ((y * 255) / height) as u8;
            let b = (((x + y) * 127) / (width + height)) as u8;
            data.push(r);
            data.push(g);
            data.push(b);
        }
    }
    data
}

fn benchmark_jpeg_encoding(c: &mut Criterion) {
    let mut group = c.benchmark_group("jpeg_encoding");

    let sizes = [
        (640, 480, "VGA"),
        (1280, 720, "HD"),
        (1920, 1080, "Full HD"),
    ];

    for (width, height, label) in sizes {
        let pixels = gradient_pixels(width, height);
        let pixel_count = (width * height) as u64;

        group.throughput(Throughput::Elements(pixel_count));

        group.bench_with_input(BenchmarkId::new("rgb", label), &pixels, |b, pixels| {
            b.iter(|| pixels_to_jpeg(black_box(pixels), black_box(width), black_box(height), black_box(bridge::ColorFormat::RGB)));
        });

        group.bench_with_input(BenchmarkId::new("bgr", label), &pixels, |b, pixels| {
            b.iter(|| pixels_to_jpeg(black_box(pixels), black_box(width), black_box(height), black_box(bridge::ColorFormat::BGR)));
        });
    }

    group.finish();
}

criterion_group!(benches, benchmark_jpeg_encoding);
criterion_main!(benches);
