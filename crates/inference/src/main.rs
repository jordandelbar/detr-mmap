use bridge::{FrameWriter, MmapReader};
use inference::{
    backend::{ort::OrtBackend, InferenceBackend},
    postprocessing::{build_detection_flatbuffer, parse_detections},
    preprocessing::preprocess_frame,
};
use ndarray::Array;
use std::thread;
use std::time::Duration;

const MODEL_PATH: &str = "/models/model.onnx";
const FRAME_MMAP_PATH: &str = "/dev/shm/bridge_frame_buffer";
const DETECTION_MMAP_PATH: &str = "/dev/shm/bridge_detection_buffer";
const DETECTION_MMAP_SIZE: usize = 1024 * 1024; // 1MB
const INPUT_SIZE: (u32, u32) = (640, 640);
const POLL_INTERVAL_MS: u64 = 100; // ~10 FPS for inference

fn main() -> anyhow::Result<()> {
    println!("Inference service starting (CPU mode)...");
    println!("Model: {}", MODEL_PATH);

    println!("Initializing inference backend...");
    let mut backend = OrtBackend::load_model(MODEL_PATH)?;
    println!("Model loaded successfully!");

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
            Array::from_shape_vec((1, 2), vec![INPUT_SIZE.1 as i64, INPUT_SIZE.0 as i64])?
                .into_dyn();

        let output = backend.infer(&preprocessed, &orig_sizes)?;

        let detections = parse_detections(
            &output.labels.view(),
            &output.boxes.view(),
            &output.scores.view(),
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
