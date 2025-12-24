use image::{ImageBuffer, Rgb};
use ndarray::{Array, IxDyn};

const INPUT_SIZE: (u32, u32) = (640, 640);

pub fn preprocess_frame(
    pixels: flatbuffers::Vector<u8>,
    width: u32,
    height: u32,
    format: schema::ColorFormat,
) -> anyhow::Result<(Array<f32, IxDyn>, f32, f32, f32)> {
    tracing::trace!(
        width,
        height,
        format = ?format,
        pixel_bytes = pixels.len(),
        "Preprocessing frame dimensions"
    );

    let expected_size = (width * height * 3) as usize;

    let mut rgb_data = Vec::with_capacity(expected_size);

    match format {
        schema::ColorFormat::RGB => {
            rgb_data.extend_from_slice(pixels.bytes());
        }
        schema::ColorFormat::BGR => {
            for i in (0..pixels.len()).step_by(3) {
                let b = pixels.get(i);
                let g = pixels.get(i + 1);
                let r = pixels.get(i + 2);
                rgb_data.push(r);
                rgb_data.push(g);
                rgb_data.push(b);
            }
        }
        schema::ColorFormat::GRAY => {
            return Err(anyhow::anyhow!("Grayscale format not supported"));
        }
        _ => {
            return Err(anyhow::anyhow!("Unknown color format"));
        }
    }

    if rgb_data.len() != expected_size {
        return Err(anyhow::anyhow!(
            "Buffer size mismatch: expected {} bytes for {}x{} RGB, got {} bytes",
            expected_size,
            width,
            height,
            rgb_data.len()
        ));
    }

    let img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_raw(width, height, rgb_data)
        .ok_or_else(|| anyhow::anyhow!("Failed to create image buffer"))?;

    let scale = (INPUT_SIZE.0 as f32 / width as f32).min(INPUT_SIZE.1 as f32 / height as f32);
    let new_width = (width as f32 * scale) as u32;
    let new_height = (height as f32 * scale) as u32;

    let resized = image::imageops::resize(
        &img,
        new_width,
        new_height,
        image::imageops::FilterType::Triangle,
    );

    let mut letterboxed =
        ImageBuffer::from_pixel(INPUT_SIZE.0, INPUT_SIZE.1, Rgb([114u8, 114u8, 114u8]));
    let offset_x = (INPUT_SIZE.0 - new_width) / 2;
    let offset_y = (INPUT_SIZE.1 - new_height) / 2;
    image::imageops::overlay(&mut letterboxed, &resized, offset_x as i64, offset_y as i64);

    let mut input = Array::zeros(IxDyn(&[1, 3, INPUT_SIZE.1 as usize, INPUT_SIZE.0 as usize]));
    for y in 0..INPUT_SIZE.1 {
        for x in 0..INPUT_SIZE.0 {
            let pixel = letterboxed.get_pixel(x, y);
            input[[0, 0, y as usize, x as usize]] = pixel[0] as f32 / 255.0;
            input[[0, 1, y as usize, x as usize]] = pixel[1] as f32 / 255.0;
            input[[0, 2, y as usize, x as usize]] = pixel[2] as f32 / 255.0;
        }
    }

    Ok((input, scale, offset_x as f32, offset_y as f32))
}
