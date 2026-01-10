use crate::config::GatewayConfig;
use crate::state::{FrameMessage, FramePacket};
use bridge::{DetectionReader, FrameReader, FrameSemaphore};
use image::{ImageBuffer, RgbImage};
use std::io::Cursor;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::broadcast;
use tokio::time;

pub async fn poll_buffers(
    config: GatewayConfig,
    tx: Arc<broadcast::Sender<FramePacket>>,
) -> anyhow::Result<()> {
    let mut frame_reader = loop {
        match FrameReader::build(&config.frame_mmap_path) {
            Ok(reader) => {
                tracing::info!("Frame buffer connected");
                break reader;
            }
            Err(_) => {
                tracing::debug!("Waiting for frame buffer...");
                time::sleep(Duration::from_millis(500)).await;
            }
        }
    };

    let mut detection_reader = loop {
        match DetectionReader::new(&config.detection_mmap_path) {
            Ok(reader) => {
                tracing::info!("Detection buffer connected");
                break reader;
            }
            Err(_) => {
                tracing::debug!("Waiting for detection buffer...");
                time::sleep(Duration::from_millis(500)).await;
            }
        }
    };

    tracing::info!("Opening gateway frame synchronization semaphore");
    let frame_semaphore = loop {
        match FrameSemaphore::open("/bridge_frame_gateway") {
            Ok(sem) => {
                tracing::info!("Gateway semaphore connected successfully");
                break Arc::new(sem);
            }
            Err(_) => {
                tracing::debug!("Waiting for gateway semaphore...");
                time::sleep(Duration::from_millis(500)).await;
            }
        }
    };

    tracing::info!("Starting event-driven buffer processing (synchronized to camera)");

    loop {
        // Wait for frame ready signal in blocking task
        let sem = frame_semaphore.clone();
        let wait_result = tokio::task::spawn_blocking(move || sem.wait()).await;

        if let Err(e) = wait_result {
            tracing::error!(error = %e, "Semaphore wait task failed");
            time::sleep(Duration::from_millis(100)).await;
            continue;
        }

        if let Err(e) = wait_result.unwrap() {
            tracing::error!(error = %e, "Semaphore wait failed");
            time::sleep(Duration::from_millis(100)).await;
            continue;
        }

        let frame_seq = frame_reader.current_sequence();
        let detection_seq = detection_reader.current_sequence();

        let frame = match frame_reader.get_frame() {
            Ok(Some(f)) => f,
            Ok(None) => {
                continue;
            }
            Err(e) => {
                tracing::error!(error = %e, sequence = frame_seq, "Failed to read frame buffer - skipping");
                frame_reader.mark_read();
                continue;
            }
        };

        let frame_num = frame.frame_number();
        let timestamp_ns = frame.timestamp_ns();
        let width = frame.width();
        let height = frame.height();

        let jpeg_data = if let Some(pixels) = frame.pixels() {
            let format = frame.format();
            let pixel_bytes = pixels.bytes();

            // Validate pixel data size
            let expected_size = (width * height * 3) as usize; // RGB = 3 bytes per pixel
            if pixel_bytes.len() < expected_size && format != schema::ColorFormat::GRAY {
                tracing::error!(
                    expected = expected_size,
                    actual = pixel_bytes.len(),
                    "Pixel buffer size mismatch - skipping JPEG encoding"
                );
                Vec::new()
            } else {
                match pixels_to_jpeg(pixel_bytes, width, height, format) {
                    Ok(data) => data,
                    Err(e) => {
                        tracing::error!("Image encoding error: {}", e);
                        Vec::new()
                    }
                }
            }
        } else {
            Vec::new()
        };

        let (detections, status) = if detection_seq > 0 {
            // Use DetectionReader to safely deserialize detection buffer
            match detection_reader.get_detections() {
                Ok(Some(dets)) => {
                    let status = if !jpeg_data.is_empty() {
                        "complete"
                    } else {
                        "detection_only"
                    };

                    (Some(dets), status.to_string())
                }
                Ok(None) => {
                    // No detections available
                    (None, "frame_only".to_string())
                }
                Err(e) => {
                    tracing::error!(
                        error = %e,
                        sequence = detection_seq,
                        "Failed to deserialize detection buffer - skipping detections"
                    );
                    // Continue with frame-only data
                    (None, "frame_only".to_string())
                }
            }
        } else {
            (None, "frame_only".to_string())
        };

        let metadata = FrameMessage {
            frame_number: frame_num,
            timestamp_ns,
            width,
            height,
            detections,
            status: status.clone(),
        };

        let packet = FramePacket {
            metadata,
            jpeg_data,
        };

        frame_reader.mark_read();
        detection_reader.mark_read();

        let det_count = packet
            .metadata
            .detections
            .as_ref()
            .map(|d| d.len())
            .unwrap_or(0);

        tracing::debug!(
            frame_number = packet.metadata.frame_number,
            detections = det_count,
            status = packet.metadata.status,
            "Frame processed"
        );

        let _ = tx.send(packet);
    }
}

fn pixels_to_jpeg(
    pixel_data: &[u8],
    width: u32,
    height: u32,
    format: schema::ColorFormat,
) -> anyhow::Result<Vec<u8>> {
    let rgb_data = match format {
        schema::ColorFormat::RGB => {
            // Already RGB, use directly
            pixel_data.to_vec()
        }
        schema::ColorFormat::BGR => {
            // Convert BGR to RGB
            let mut rgb_data = Vec::with_capacity(pixel_data.len());
            for chunk in pixel_data.chunks_exact(3) {
                rgb_data.push(chunk[2]); // R
                rgb_data.push(chunk[1]); // G
                rgb_data.push(chunk[0]); // B
            }
            rgb_data
        }
        schema::ColorFormat::GRAY => {
            return Err(anyhow::anyhow!(
                "Grayscale format not supported for JPEG encoding"
            ));
        }
        _ => {
            return Err(anyhow::anyhow!("Unknown color format"));
        }
    };

    let img: RgbImage = ImageBuffer::from_raw(width, height, rgb_data)
        .ok_or_else(|| anyhow::anyhow!("Failed to create image from raw data"))?;

    let mut jpeg_bytes = Cursor::new(Vec::new());
    img.write_to(&mut jpeg_bytes, image::ImageFormat::Jpeg)?;

    Ok(jpeg_bytes.into_inner())
}
