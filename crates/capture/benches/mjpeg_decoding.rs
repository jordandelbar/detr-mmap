use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};

/// Create a JPEG encoded test image with a gradient pattern
fn create_test_jpeg(width: u32, height: u32, quality: u8) -> Vec<u8> {
    use image::{ImageEncoder, codecs::jpeg::JpegEncoder};

    let size = (width * height * 3) as usize;
    let mut pixels = Vec::with_capacity(size);

    for y in 0..height {
        for x in 0..width {
            let r = ((x * 255) / width) as u8;
            let g = ((y * 255) / height) as u8;
            let b = (((x + y) * 127) / (width + height)) as u8;
            pixels.push(r);
            pixels.push(g);
            pixels.push(b);
        }
    }

    let mut jpeg_data = Vec::new();
    let encoder = JpegEncoder::new_with_quality(&mut jpeg_data, quality);
    encoder
        .write_image(&pixels, width, height, image::ExtendedColorType::Rgb8)
        .expect("Failed to encode test JPEG");

    jpeg_data
}

fn benchmark_mjpeg_decoding(c: &mut Criterion) {
    let mut group = c.benchmark_group("mjpeg_decoding");

    let sizes = [
        (640, 480, "VGA"),
        (1280, 720, "HD"),
        (1920, 1080, "Full HD"),
    ];

    for (width, height, label) in sizes {
        let jpeg_data = create_test_jpeg(width, height, 85);
        let pixel_count = (width * height) as u64;

        group.throughput(Throughput::Elements(pixel_count));

        group.bench_with_input(
            BenchmarkId::new("current", label),
            &jpeg_data,
            |b, jpeg| {
                b.iter(|| capture::mjpeg_to_rgb(black_box(jpeg)))
            },
        );
    }

    group.finish();
}

criterion_group!(benches, benchmark_mjpeg_decoding);
criterion_main!(benches);
