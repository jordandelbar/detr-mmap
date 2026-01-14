use crate::state::{FrameMessage, FramePacket};
use bridge::{BridgeSemaphore, DetectionReader, FrameReader, SemaphoreType};
use common::wait_for_resource_async;
use image::{ImageBuffer, RgbImage};
use std::io::Cursor;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::broadcast;
use tokio::time;

/// Frame data extracted from shared memory
struct FrameData {
    frame_number: u64,
    timestamp_ns: u64,
    width: u32,
    height: u32,
    pixel_data: Vec<u8>,
    format: bridge::ColorFormat,
}

/// Detection data with status information
struct DetectionData {
    detections: Vec<bridge::Detection>,
    has_jpeg: bool,
}

pub struct BufferPoller {
    frame_reader: FrameReader,
    detection_reader: DetectionReader,
    frame_semaphore: Arc<BridgeSemaphore>,
    tx: Arc<broadcast::Sender<FramePacket>>,
}

const POLL_INTERVAL_MS: u64 = 500;

impl BufferPoller {
    /// Build a new BufferPoller by connecting to shared memory buffers with retries
    pub async fn build(tx: Arc<broadcast::Sender<FramePacket>>) -> anyhow::Result<Self> {
        let frame_reader = wait_for_resource_async(FrameReader::build, POLL_INTERVAL_MS, "Frame buffer").await;
        let detection_reader = wait_for_resource_async(DetectionReader::build, POLL_INTERVAL_MS, "Detection buffer").await;
        let frame_semaphore = Arc::new(
            wait_for_resource_async(
                || BridgeSemaphore::open(SemaphoreType::FrameCaptureToGateway),
                POLL_INTERVAL_MS,
                "Gateway semaphore",
            )
            .await,
        );

        Ok(Self {
            frame_reader,
            detection_reader,
            frame_semaphore,
            tx,
        })
    }

    /// Main polling loop
    pub async fn run(mut self) -> anyhow::Result<()> {
        tracing::info!("Starting event-driven buffer processing (synchronized to camera)");

        loop {
            // Wait for frame ready signal
            if let Err(e) = self.wait_for_frame().await {
                tracing::error!(error = %e, "Frame wait failed");
                time::sleep(Duration::from_millis(100)).await;
                continue;
            }

            // Read current frame
            let frame_data = match self.read_current_frame() {
                Ok(data) => data,
                Err(e) => {
                    tracing::error!(error = %e, "Failed to read frame - skipping");
                    continue;
                }
            };

            // Encode frame to JPEG
            let jpeg_data = self.encode_to_jpeg(&frame_data);

            // Read detections if available
            let detection_data = self.read_detections(!jpeg_data.is_empty());

            // Build and broadcast packet
            let packet = self.build_packet(frame_data, jpeg_data, detection_data);
            self.broadcast_packet(packet);

            // Mark buffers as read
            self.frame_reader.mark_read();
            self.detection_reader.mark_read();
        }
    }

    /// Wait for frame ready signal from camera
    async fn wait_for_frame(&self) -> anyhow::Result<()> {
        let sem = self.frame_semaphore.clone();
        let wait_result = tokio::task::spawn_blocking(move || sem.wait()).await;

        match wait_result {
            Ok(Ok(())) => Ok(()),
            Ok(Err(e)) => {
                anyhow::bail!("Semaphore wait failed: {}", e)
            }
            Err(e) => {
                anyhow::bail!(
                    "Semaphore wait task failed (task panicked or cancelled): {}",
                    e
                )
            }
        }
    }

    /// Read and validate current frame from shared memory
    fn read_current_frame(&mut self) -> anyhow::Result<FrameData> {
        let frame_seq = self.frame_reader.current_sequence();

        let frame = match self.frame_reader.get_frame() {
            Ok(Some(f)) => f,
            Ok(None) => {
                anyhow::bail!("No frame available")
            }
            Err(e) => {
                // Mark as read even on error to prevent getting stuck
                self.frame_reader.mark_read();
                anyhow::bail!(
                    "Failed to read frame buffer (sequence {}): {}",
                    frame_seq,
                    e
                )
            }
        };

        let frame_number = frame.frame_number();
        let timestamp_ns = frame.timestamp_ns();
        let width = frame.width();
        let height = frame.height();
        let format = frame.format();

        // Extract pixel data from frame
        let pixel_data = if let Some(pixels) = frame.pixels() {
            pixels.bytes().to_vec()
        } else {
            Vec::new()
        };

        Ok(FrameData {
            frame_number,
            timestamp_ns,
            width,
            height,
            pixel_data,
            format,
        })
    }

    /// Read detections from shared memory if available
    fn read_detections(&mut self, has_jpeg: bool) -> Option<DetectionData> {
        let detection_seq = self.detection_reader.current_sequence();

        if detection_seq == 0 {
            return None;
        }

        match self.detection_reader.get_detections() {
            Ok(Some(detections)) => Some(DetectionData {
                detections,
                has_jpeg,
            }),
            Ok(None) => None,
            Err(e) => {
                tracing::error!(
                    error = %e,
                    sequence = detection_seq,
                    "Failed to deserialize detection buffer - skipping detections"
                );
                None
            }
        }
    }

    /// Encode frame pixels to JPEG
    fn encode_to_jpeg(&self, frame_data: &FrameData) -> Vec<u8> {
        if frame_data.pixel_data.is_empty() {
            return Vec::new();
        }

        // Validate pixel data size
        let expected_size = (frame_data.width * frame_data.height * 3) as usize;
        if frame_data.pixel_data.len() < expected_size
            && frame_data.format != bridge::ColorFormat::GRAY
        {
            tracing::error!(
                expected = expected_size,
                actual = frame_data.pixel_data.len(),
                "Pixel buffer size mismatch - skipping JPEG encoding"
            );
            return Vec::new();
        }

        match pixels_to_jpeg(
            &frame_data.pixel_data,
            frame_data.width,
            frame_data.height,
            frame_data.format,
        ) {
            Ok(data) => data,
            Err(e) => {
                tracing::error!("Image encoding error: {}", e);
                Vec::new()
            }
        }
    }

    /// Build packet for broadcast
    fn build_packet(
        &self,
        frame_data: FrameData,
        jpeg_data: Vec<u8>,
        detection_data: Option<DetectionData>,
    ) -> FramePacket {
        let (detections, status) = match detection_data {
            Some(DetectionData {
                detections,
                has_jpeg,
            }) => {
                let status = if has_jpeg {
                    "complete"
                } else {
                    "detection_only"
                };
                (Some(detections), status.to_string())
            }
            None => (None, "frame_only".to_string()),
        };

        let metadata = FrameMessage {
            frame_number: frame_data.frame_number,
            timestamp_ns: frame_data.timestamp_ns,
            width: frame_data.width,
            height: frame_data.height,
            detections,
            status,
        };

        FramePacket {
            metadata,
            jpeg_data,
        }
    }

    /// Broadcast packet to WebSocket clients
    fn broadcast_packet(&self, packet: FramePacket) {
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

        let _ = self.tx.send(packet);
    }
}

fn pixels_to_jpeg(
    pixel_data: &[u8],
    width: u32,
    height: u32,
    format: bridge::ColorFormat,
) -> anyhow::Result<Vec<u8>> {
    let rgb_data = match format {
        bridge::ColorFormat::RGB => {
            // Already RGB, use directly
            pixel_data.to_vec()
        }
        bridge::ColorFormat::BGR => {
            // Convert BGR to RGB
            let mut rgb_data = Vec::with_capacity(pixel_data.len());
            for chunk in pixel_data.chunks_exact(3) {
                rgb_data.push(chunk[2]); // R
                rgb_data.push(chunk[1]); // G
                rgb_data.push(chunk[0]); // B
            }
            rgb_data
        }
        bridge::ColorFormat::GRAY => {
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
