use opencv::{
    prelude::*,
    videoio::{self, CAP_ANY, VideoCapture},
};
use schema::{ColorFormat, FrameArgs, FrameWriter};
use std::time::{SystemTime, UNIX_EPOCH};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Starting camera capture from /dev/video0...");

    // Open the camera (device 0)
    let mut cam = VideoCapture::new(0, CAP_ANY)?;

    if !VideoCapture::is_opened(&cam)? {
        return Err("Failed to open camera at /dev/video0".into());
    }

    println!("Camera opened successfully!");

    let width = cam.get(videoio::CAP_PROP_FRAME_WIDTH)?;
    let height = cam.get(videoio::CAP_PROP_FRAME_HEIGHT)?;
    let fps = cam.get(videoio::CAP_PROP_FPS)?;

    println!("Camera properties:");
    println!("  Resolution: {}x{}", width, height);
    println!("  FPS: {}", fps);

    let mmap_size = 8 * 1024 * 1024;
    let mmap_path = "/dev/shm/bridge_frame_buffer";
    let mut writer = FrameWriter::new(mmap_path, mmap_size)?;
    println!(
        "Created mmap at {} ({} MB)",
        mmap_path,
        mmap_size / 1024 / 1024
    );
    println!("\nCapturing frames (Ctrl+C to stop)...\n");

    let mut frame = Mat::default();
    let mut frame_count = 0u64;

    loop {
        cam.read(&mut frame)?;

        if frame.empty() {
            eprintln!("Warning: Empty frame received");
            continue;
        }

        let size = frame.size()?;
        let frame_width = size.width as u32;
        let frame_height = size.height as u32;
        let channels = frame.channels() as u8;

        let pixel_data = frame.data_bytes()?;

        let timestamp_ns = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos() as u64;

        let mut builder = flatbuffers::FlatBufferBuilder::new();
        let pixels_vec = builder.create_vector(pixel_data);

        let frame_fb = schema::Frame::create(
            &mut builder,
            &FrameArgs {
                frame_number: frame_count,
                timestamp_ns,
                camera_id: 0,
                width: frame_width,
                height: frame_height,
                channels,
                format: ColorFormat::BGR,
                pixels: Some(pixels_vec),
            },
        );

        builder.finish(frame_fb, None);
        let data = builder.finished_data();

        writer.write(data)?;

        frame_count += 1;

        println!("Frame #{}", frame_count);
        println!("  Size: {}x{}", frame_width, frame_height);
        println!("  Channels: {}", channels);
        println!("  Pixel data size: {} bytes", pixel_data.len());
        println!("  FlatBuffer size: {} bytes", data.len());
        println!("  Written to: {}", mmap_path);
        println!();

        std::thread::sleep(std::time::Duration::from_millis(1000));
    }
}
