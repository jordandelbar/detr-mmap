use bridge::{FrameWriter, Semaphore};
use flatbuffers::FlatBufferBuilder;
use image::{ImageBuffer, Rgb};
use memmap2::Mmap;
use ndarray::{Array, IxDyn};
use ort::{
    session::{Session, builder::GraphOptimizationLevel},
    value::TensorRef,
};
use std::fs::File;

const MODEL_PATH: &str = "/models/model.onnx";
const FRAME_MMAP_PATH: &str = "/dev/shm/bridge_frame_buffer";
const DETECTION_MMAP_PATH: &str = "/dev/shm/bridge_detection_buffer";
const DETECTION_MMAP_SIZE: usize = 1024 * 1024; // 1MB
const CONFIDENCE_THRESHOLD: f32 = 0.5;
const INPUT_SIZE: (u32, u32) = (640, 640);

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

    println!("\nReading frames from: {}", FRAME_MMAP_PATH);
    let file = File::open(FRAME_MMAP_PATH)?;
    let mmap = unsafe { Mmap::map(&file)? };
    println!("Frame mmap size: {} MB", mmap.len() / 1024 / 1024);

    println!("Creating detection buffer at: {}", DETECTION_MMAP_PATH);
    let mut detection_writer = FrameWriter::new(DETECTION_MMAP_PATH, DETECTION_MMAP_SIZE)?;
    println!(
        "Detection mmap size: {} MB\n",
        DETECTION_MMAP_SIZE / 1024 / 1024
    );

    let frame_writer_sem = Semaphore::open("/bridge_writer_sem")?;
    let frame_reader_sem = Semaphore::open("/bridge_reader_sem")?;

    let detection_writer_sem = Semaphore::new("/bridge_detection_writer_sem", 1)?;
    let detection_reader_sem = Semaphore::new("/bridge_detection_reader_sem", 0)?;
    println!("Semaphores initialized\n");

    let mut last_frame_num = 0u64;
    let mut total_detections = 0usize;

    loop {
        frame_reader_sem.wait()?;

        let frame = flatbuffers::root::<schema::Frame>(&mmap)?;
        let frame_num = frame.frame_number();
        let timestamp_ns = frame.timestamp_ns();
        let camera_id = frame.camera_id();
        let width = frame.width();
        let height = frame.height();

        if last_frame_num > 0 && frame_num != last_frame_num + 1 {
            let dropped = frame_num - last_frame_num - 1;
            eprintln!("⚠ Dropped {} frame(s)", dropped);
        }
        last_frame_num = frame_num;

        let pixels = frame
            .pixels()
            .ok_or_else(|| anyhow::anyhow!("No pixel data"))?;

        let preprocessed = preprocess_frame(pixels, width, height)?;

        let orig_sizes = Array::from_shape_vec((1, 2), vec![height as i64, width as i64])?;

        let outputs = session.run(ort::inputs![
            "images" => TensorRef::from_array_view(&preprocessed)?,
            "orig_target_sizes" => TensorRef::from_array_view(&orig_sizes)?
        ])?;

        let labels = outputs["labels"].try_extract_array::<i64>()?; // Class IDs as integers
        let boxes = outputs["boxes"].try_extract_array::<f32>()?;
        let scores = outputs["scores"].try_extract_array::<f32>()?;

        eprintln!(
            "Output shapes - labels: {:?}, boxes: {:?}, scores: {:?}",
            labels.shape(),
            boxes.shape(),
            scores.shape()
        );

        let detections =
            parse_detections(&labels.view(), &boxes.view(), &scores.view(), width, height)?;

        let detection_buffer =
            build_detection_flatbuffer(frame_num, timestamp_ns, camera_id, &detections)?;

        detection_writer_sem.wait()?;
        detection_writer.write(&detection_buffer)?;
        detection_reader_sem.post()?;

        frame_writer_sem.post()?;

        total_detections += detections.len();
        println!(
            "Frame {}: {} detections | Total: {}",
            frame_num,
            detections.len(),
            total_detections
        );
    }
}

fn preprocess_frame(
    pixels: flatbuffers::Vector<u8>,
    width: u32,
    height: u32,
) -> anyhow::Result<Array<f32, IxDyn>> {
    let mut rgb_data = Vec::with_capacity((width * height * 3) as usize);
    for i in (0..pixels.len()).step_by(3) {
        let b = pixels.get(i);
        let g = pixels.get(i + 1);
        let r = pixels.get(i + 2);
        rgb_data.push(r);
        rgb_data.push(g);
        rgb_data.push(b);
    }

    let img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_raw(width, height, rgb_data)
        .ok_or_else(|| anyhow::anyhow!("Failed to create image buffer"))?;

    let resized = image::imageops::resize(
        &img,
        INPUT_SIZE.0,
        INPUT_SIZE.1,
        image::imageops::FilterType::Triangle,
    );

    let mut input = Array::zeros(IxDyn(&[1, 3, INPUT_SIZE.1 as usize, INPUT_SIZE.0 as usize]));
    for y in 0..INPUT_SIZE.1 {
        for x in 0..INPUT_SIZE.0 {
            let pixel = resized.get_pixel(x, y);
            input[[0, 0, y as usize, x as usize]] = pixel[0] as f32 / 255.0;
            input[[0, 1, y as usize, x as usize]] = pixel[1] as f32 / 255.0;
            input[[0, 2, y as usize, x as usize]] = pixel[2] as f32 / 255.0;
        }
    }

    Ok(input)
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
) -> anyhow::Result<Vec<Detection>> {
    let mut detections = Vec::new();

    let num_queries = labels.shape()[1];

    for i in 0..num_queries {
        let class_id = labels[[0, i]];
        let confidence = scores[[0, i]];

        if confidence < CONFIDENCE_THRESHOLD {
            continue;
        }

        let cx = boxes[[0, i, 0]];
        let cy = boxes[[0, i, 1]];
        let w = boxes[[0, i, 2]];
        let h = boxes[[0, i, 3]];

        let x1 = ((cx - w / 2.0) * orig_width as f32).max(0.0);
        let y1 = ((cy - h / 2.0) * orig_height as f32).max(0.0);
        let x2 = ((cx + w / 2.0) * orig_width as f32).min(orig_width as f32);
        let y2 = ((cy + h / 2.0) * orig_height as f32).min(orig_height as f32);

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
