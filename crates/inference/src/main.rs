use bridge::{FrameWriter, MmapReader};
use flatbuffers::FlatBufferBuilder;
use image::{ImageBuffer, Rgb};
use ndarray::{Array, IxDyn};
use ort::{
    session::{Session, builder::GraphOptimizationLevel},
    value::TensorRef,
};
use std::thread;
use std::time::Duration;

const MODEL_PATH: &str = "/models/model.onnx";
const FRAME_MMAP_PATH: &str = "/dev/shm/bridge_frame_buffer";
const DETECTION_MMAP_PATH: &str = "/dev/shm/bridge_detection_buffer";
const DETECTION_MMAP_SIZE: usize = 1024 * 1024; // 1MB
const CONFIDENCE_THRESHOLD: f32 = 0.5;
const INPUT_SIZE: (u32, u32) = (640, 640);
const POLL_INTERVAL_MS: u64 = 100; // ~10 FPS for inference

fn main() -> anyhow::Result<()> {
    println!("Inference service starting (CPU mode)...");
    println!("Model: {}", MODEL_PATH);

    println!("Initializing ONNX Runtime...");
    let mut session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .commit_from_file(MODEL_PATH)?;

    println!("Model loaded successfully!");
    println!(
        "Inputs: {:?}",
        session.inputs.iter().map(|i| &i.name).collect::<Vec<_>>()
    );
    println!(
        "Outputs: {:?}",
        session.outputs.iter().map(|o| &o.name).collect::<Vec<_>>()
    );

    println!("\nWaiting for frame buffer at: {}", FRAME_MMAP_PATH);
    let mut frame_reader = loop {
        match MmapReader::new(FRAME_MMAP_PATH) {
            Ok(reader) => break reader,
            Err(_) => {
                thread::sleep(Duration::from_millis(100));
            }
        }
    };
    println!("Frame buffer connected!");

    println!("Creating detection buffer at: {}", DETECTION_MMAP_PATH);
    let mut detection_writer = FrameWriter::new(DETECTION_MMAP_PATH, DETECTION_MMAP_SIZE)?;
    println!(
        "Detection buffer ready ({} MB)\n",
        DETECTION_MMAP_SIZE / 1024 / 1024
    );

    println!("Polling for frames ({}ms interval)...\n", POLL_INTERVAL_MS);

    let mut total_detections = 0usize;
    let mut frames_processed = 0u64;

    loop {
        if !frame_reader.has_new_data() {
            thread::sleep(Duration::from_millis(POLL_INTERVAL_MS));
            continue;
        }

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

        let (preprocessed, scale, offset_x, offset_y) = preprocess_frame(pixels, width, height, format)?;

        let orig_sizes =
            Array::from_shape_vec((1, 2), vec![INPUT_SIZE.1 as i64, INPUT_SIZE.0 as i64])?;

        let outputs = session.run(ort::inputs![
            "images" => TensorRef::from_array_view(&preprocessed)?,
            "orig_target_sizes" => TensorRef::from_array_view(&orig_sizes)?
        ])?;

        let labels = outputs["labels"].try_extract_array::<i64>()?;
        let boxes = outputs["boxes"].try_extract_array::<f32>()?;
        let scores = outputs["scores"].try_extract_array::<f32>()?;

        let detections = parse_detections(
            &labels.view(),
            &boxes.view(),
            &scores.view(),
            width,
            height,
            scale,
            offset_x,
            offset_y,
        )?;

        let detection_buffer =
            build_detection_flatbuffer(frame_num, timestamp_ns, camera_id, &detections)?;

        detection_writer.write(&detection_buffer)?;

        frame_reader.mark_read();
        frames_processed += 1;
        total_detections += detections.len();

        println!(
            "Frame {} (seq: {}): {} detections | Processed: {} | Total detections: {}",
            frame_num,
            frame_reader.last_sequence(),
            detections.len(),
            frames_processed,
            total_detections
        );
    }
}

fn preprocess_frame(
    pixels: flatbuffers::Vector<u8>,
    width: u32,
    height: u32,
    format: schema::ColorFormat,
) -> anyhow::Result<(Array<f32, IxDyn>, f32, f32, f32)> {
    println!("Frame dimensions: {}x{}, format: {:?}, pixel bytes: {}", width, height, format, pixels.len());

    let expected_size = (width * height * 3) as usize;
    println!("Expected RGB buffer size: {}", expected_size);

    let mut rgb_data = Vec::with_capacity(expected_size);

    match format {
        schema::ColorFormat::RGB => {
            // Already RGB, just copy
            rgb_data.extend_from_slice(pixels.bytes());
        }
        schema::ColorFormat::BGR => {
            // Convert BGR to RGB
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

    println!("Actual RGB buffer size: {}", rgb_data.len());

    if rgb_data.len() != expected_size {
        return Err(anyhow::anyhow!(
            "Buffer size mismatch: expected {} bytes for {}x{} RGB, got {} bytes",
            expected_size, width, height, rgb_data.len()
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

struct Detection {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    confidence: f32,
    class_id: u32,
}

fn parse_detections(
    labels: &ndarray::ArrayViewD<i64>,
    boxes: &ndarray::ArrayViewD<f32>,
    scores: &ndarray::ArrayViewD<f32>,
    orig_width: u32,
    orig_height: u32,
    scale: f32,
    offset_x: f32,
    offset_y: f32,
) -> anyhow::Result<Vec<Detection>> {
    let mut detections = Vec::new();

    let num_queries = labels.shape()[1];

    for i in 0..num_queries {
        let class_id = labels[[0, i]];
        let confidence = scores[[0, i]];

        if confidence < CONFIDENCE_THRESHOLD {
            continue;
        }

        let x1 = ((boxes[[0, i, 0]] - offset_x) / scale)
            .max(0.0)
            .min(orig_width as f32);
        let y1 = ((boxes[[0, i, 1]] - offset_y) / scale)
            .max(0.0)
            .min(orig_height as f32);
        let x2 = ((boxes[[0, i, 2]] - offset_x) / scale)
            .max(0.0)
            .min(orig_width as f32);
        let y2 = ((boxes[[0, i, 3]] - offset_y) / scale)
            .max(0.0)
            .min(orig_height as f32);

        detections.push(Detection {
            x1,
            y1,
            x2,
            y2,
            confidence,
            class_id: class_id as u32,
        });
    }

    Ok(detections)
}

fn build_detection_flatbuffer(
    frame_number: u64,
    timestamp_ns: u64,
    camera_id: u32,
    detections: &[Detection],
) -> anyhow::Result<Vec<u8>> {
    let mut builder = FlatBufferBuilder::new();

    let bbox_vec: Vec<_> = detections
        .iter()
        .map(|d| {
            schema::BoundingBox::create(
                &mut builder,
                &schema::BoundingBoxArgs {
                    x1: d.x1,
                    y1: d.y1,
                    x2: d.x2,
                    y2: d.y2,
                    confidence: d.confidence,
                    class_id: d.class_id,
                },
            )
        })
        .collect();

    let detections_offset = builder.create_vector(&bbox_vec);

    let detection_result = schema::DetectionResult::create(
        &mut builder,
        &schema::DetectionResultArgs {
            frame_number,
            timestamp_ns,
            camera_id,
            detections: Some(detections_offset),
        },
    );

    builder.finish(detection_result, None);
    Ok(builder.finished_data().to_vec())
}
