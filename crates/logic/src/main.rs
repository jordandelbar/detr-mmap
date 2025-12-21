use bridge::{MmapReader, Semaphore};

const DETECTION_MMAP_PATH: &str = "/dev/shm/bridge_detection_buffer";

fn main() -> anyhow::Result<()> {
    println!("Logic service starting...");
    println!("Reading detections from: {}\n", DETECTION_MMAP_PATH);

    let detection_reader = MmapReader::new(DETECTION_MMAP_PATH)?;
    println!("Detection mmap opened successfully!");

    let detection_writer_sem = Semaphore::open("/bridge_detection_writer_sem")?;
    let detection_reader_sem = Semaphore::open("/bridge_detection_reader_sem")?;
    println!("Connected to detection semaphores\n");

    let mut total_frames = 0u64;
    let mut total_detections = 0usize;

    loop {
        detection_reader_sem.wait()?;

        let detection_result =
            flatbuffers::root::<schema::DetectionResult>(detection_reader.buffer())?;

        let frame_num = detection_result.frame_number();
        let timestamp_ns = detection_result.timestamp_ns();
        let camera_id = detection_result.camera_id();

        total_frames += 1;

        if let Some(detections) = detection_result.detections() {
            let num_detections = detections.len();
            total_detections += num_detections;

            println!("=== Frame {} (Camera {}) ===", frame_num, camera_id);
            println!("Timestamp: {} ns", timestamp_ns);
            println!("Detections: {}", num_detections);

            for (i, detection) in detections.iter().enumerate() {
                println!(
                    "  [{}/{}] Class {}: {:.2}% @ ({:.1}, {:.1}) - ({:.1}, {:.1})",
                    i + 1,
                    num_detections,
                    detection.class_id(),
                    detection.confidence() * 100.0,
                    detection.x1(),
                    detection.y1(),
                    detection.x2(),
                    detection.y2()
                );
            }

            println!(
                "Total: {} frames processed, {} detections\n",
                total_frames, total_detections
            );
        } else {
            println!(
                "Frame {}: No detections (Total: {} frames, {} detections)\n",
                frame_num, total_frames, total_detections
            );
        }

        detection_writer_sem.post()?;
    }
}
