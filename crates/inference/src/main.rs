use bridge::Semaphore;
use memmap2::Mmap;
use std::fs::File;
use std::thread;
use std::time::Duration;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mmap_path = "/dev/shm/bridge_frame_buffer";
    println!("Inference service starting...");
    println!("Reading frames from: {}\n", mmap_path);

    let file = File::open(mmap_path)?;
    let mmap = unsafe { Mmap::map(&file)? };

    println!("Memory-mapped file opened successfully!");
    println!(
        "Mmap size: {} bytes ({} MB)\n",
        mmap.len(),
        mmap.len() / 1024 / 1024
    );

    let writer_sem = Semaphore::open("/bridge_writer_sem")?;
    let reader_sem = Semaphore::open("/bridge_reader_sem")?;
    println!("Connected to semaphores for synchronization\n");

    let mut last_frame_num = 0u64;

    loop {
        println!("Waiting for new frame...");
        reader_sem.wait()?;

        let frame = flatbuffers::root::<schema::Frame>(&mmap)?;

        let frame_num = frame.frame_number();

        if last_frame_num > 0 && frame_num != last_frame_num + 1 {
            let dropped = frame_num - last_frame_num - 1;
            eprintln!("Detected {} dropped frame(s)", dropped);
        }

        println!("=== New Frame Received ===");
        println!("Frame number: {}", frame_num);
        println!("Timestamp: {} ns", frame.timestamp_ns());
        println!("Camera ID: {}", frame.camera_id());
        println!("Resolution: {}x{}", frame.width(), frame.height());
        println!("Channels: {}", frame.channels());
        println!("Format: {:?}", frame.format());

        if let Some(pixels) = frame.pixels() {
            println!("Pixel data size: {} bytes", pixels.len());
            let sample_size = 30.min(pixels.len());
            let mut sample = Vec::with_capacity(sample_size);
            for i in 0..sample_size {
                sample.push(pixels.get(i));
            }
            println!("First 10 pixels (BGR): {:?}", sample);
        }

        println!();
        last_frame_num = frame_num;

        thread::sleep(Duration::from_millis(100));

        writer_sem.post()?;
        println!("Processing complete, ready for next frame\n");
    }
}
