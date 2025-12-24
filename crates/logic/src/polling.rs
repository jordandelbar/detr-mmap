use crate::config::LogicConfig;
use crate::state::{Detection, FrameMessage, FramePacket};
use bridge::MmapReader;
use image::{ImageBuffer, RgbImage};
use std::io::Cursor;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::broadcast;
use tokio::time;

pub async fn poll_buffers(
    config: LogicConfig,
    tx: Arc<broadcast::Sender<FramePacket>>,
) -> anyhow::Result<()> {
    let mut frame_reader = loop {
        match MmapReader::new(&config.frame_mmap_path) {
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
        match MmapReader::new(&config.detection_mmap_path) {
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

    tracing::info!(
        "Polling buffers at {}ms intervals",
        config.poll_interval_ms
    );

    let mut interval = time::interval(Duration::from_millis(config.poll_interval_ms));

    loop {
        interval.tick().await;

        let frame_seq = frame_reader.current_sequence();
        let detection_seq = detection_reader.current_sequence();

        if frame_seq == 0 {
            continue;
        }

        let frame = flatbuffers::root::<schema::Frame>(frame_reader.buffer())?;
        let frame_num = frame.frame_number();
        let timestamp_ns = frame.timestamp_ns();
        let width = frame.width();
        let height = frame.height();

        let jpeg_data = if let Some(pixels) = frame.pixels() {
            let format = frame.format();
            match pixels_to_jpeg(pixels.bytes(), width, height, format) {
                Ok(data) => data,
                Err(e) => {
                    tracing::error!("Image encoding error: {}", e);
                    Vec::new()
                }
            }
        } else {
            Vec::new()
        };

        let (detections, status) = if detection_seq > 0 {
            let detection =
                flatbuffers::root::<schema::DetectionResult>(detection_reader.buffer())?;
            let dets = detection.detections().map(|d| {
                d.iter()
                    .map(|det| Detection {
                        x1: det.x1(),
                        y1: det.y1(),
                        x2: det.x2(),
                        y2: det.y2(),
                        confidence: det.confidence(),
                        class_id: det.class_id(),
                    })
                    .collect()
            });

            let status = if !jpeg_data.is_empty() {
                "complete"
            } else {
                "detection_only"
            };

            (dets, status.to_string())
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
