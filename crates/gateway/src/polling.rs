use crate::state::{FrameMessage, FramePacket};
use bridge::{BridgeSemaphore, Detection, DetectionReader, FrameReader, SemaphoreType, set_trace_parent};
use common::{span, wait_for_resource_async};
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
    trace_ctx: Option<schema::TraceContext>,
}

/// Detection data with status information
struct DetectionData {
    detections: Vec<Detection>,
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
        let frame_reader =
            wait_for_resource_async(FrameReader::build, POLL_INTERVAL_MS, "Frame buffer").await;
        let detection_reader =
            wait_for_resource_async(DetectionReader::build, POLL_INTERVAL_MS, "Detection buffer")
                .await;
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

            // Create span and link to parent trace from capture service
            let span = tracing::info_span!("gateway_process_frame");
            if let Some(ref ctx) = frame_data.trace_ctx {
                set_trace_parent(ctx, &span);
            }
            let _guard = span.entered();

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
        let _s = span!("read_current_frame");

        let frame_seq = self.frame_reader.current_sequence();

        let frame = match self.frame_reader.get_frame() {
            Ok(Some(data)) => data,
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

        // Extract trace context if present (copy the 25-byte struct)
        let trace_ctx = frame.trace().copied();

        // Extract pixel data from frame
        let pixel_data = if let Some(pixels) = frame.pixels() {
            // TODO: change to Cow<'a, [u8]> or Arc<[u8]>
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
            trace_ctx,
        })
    }

    /// Read detections from shared memory if available.
    /// Converts from zero-copy FlatBuffers to owned BoundingBox for serialization.
    fn read_detections(&mut self, has_jpeg: bool) -> Option<DetectionData> {
        let _s = span!("read_detections");

        let detection_seq = self.detection_reader.current_sequence();

        if detection_seq == 0 {
            return None;
        }

        match self.detection_reader.get_detections() {
            Ok(Some(detection_result)) => {
                // Convert FlatBuffers detections to owned Detection at serialization boundary
                let detections = detection_result
                    .detections()
                    .map(|dets| dets.iter().map(|d| Detection::from(&d)).collect())
                    .unwrap_or_default();

                Some(DetectionData {
                    detections,
                    has_jpeg,
                })
            }
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
        let _s = span!("encode_to_jpeg");

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
        let _s = span!("build_packet");

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
        let _s = span!("broadcast_packet");

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

/// JPEG encoding quality (0-100)
const JPEG_QUALITY: i32 = 80;

/// Convert raw pixel data to JPEG format using turbojpeg
/// Supports RGB and BGR color formats
pub fn pixels_to_jpeg(
    pixel_data: &[u8],
    width: u32,
    height: u32,
    format: bridge::ColorFormat,
) -> anyhow::Result<Vec<u8>> {
    let _s = span!("pixels_to_jpeg");

    let pixel_format = match format {
        bridge::ColorFormat::RGB => turbojpeg::PixelFormat::RGB,
        bridge::ColorFormat::BGR => turbojpeg::PixelFormat::BGR,
        bridge::ColorFormat::GRAY => {
            return Err(anyhow::anyhow!(
                "Grayscale format not supported for JPEG encoding"
            ));
        }
        _ => {
            return Err(anyhow::anyhow!("Unknown color format"));
        }
    };

    // Validate buffer size to avoid panic in turbojpeg
    let expected_size = (width as usize) * (height as usize) * 3;
    if pixel_data.len() < expected_size {
        return Err(anyhow::anyhow!(
            "Pixel buffer too small: got {}, expected {}",
            pixel_data.len(),
            expected_size
        ));
    }

    let image = turbojpeg::Image {
        pixels: pixel_data,
        width: width as usize,
        pitch: (width * 3) as usize,
        height: height as usize,
        format: pixel_format,
    };

    let jpeg_data = turbojpeg::compress(image, JPEG_QUALITY, turbojpeg::Subsamp::Sub2x2)?;

    Ok(jpeg_data.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create test pixel data of a solid color
    fn solid_color_pixels(width: u32, height: u32, r: u8, g: u8, b: u8) -> Vec<u8> {
        let size = (width * height * 3) as usize;
        let mut data = Vec::with_capacity(size);
        for _ in 0..(width * height) {
            data.push(r);
            data.push(g);
            data.push(b);
        }
        data
    }

    #[test]
    fn rgb_passthrough_produces_valid_jpeg() {
        let pixels = solid_color_pixels(64, 64, 255, 0, 0); // Red
        let result = pixels_to_jpeg(&pixels, 64, 64, bridge::ColorFormat::RGB);

        assert!(result.is_ok(), "Error: {:?}", result.err());
        let jpeg = result.unwrap();
        // JPEG magic bytes
        assert!(jpeg.len() > 2);
        assert_eq!(&jpeg[0..2], &[0xFF, 0xD8]);
    }

    #[test]
    fn bgr_to_rgb_conversion() {
        // BGR: Blue=255, Green=0, Red=0 -> should become RGB: Red=0, Green=0, Blue=255
        let bgr_pixels = solid_color_pixels(64, 64, 0, 0, 255); // BGR order: B=0, G=0, R=255
        let result = pixels_to_jpeg(&bgr_pixels, 64, 64, bridge::ColorFormat::BGR);

        assert!(result.is_ok());
        let jpeg = result.unwrap();
        assert!(jpeg.len() > 2);
        assert_eq!(&jpeg[0..2], &[0xFF, 0xD8]);
    }

    #[test]
    fn grayscale_returns_error() {
        let pixels = vec![128u8; 64 * 64]; // Single channel
        let result = pixels_to_jpeg(&pixels, 64, 64, bridge::ColorFormat::GRAY);

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Grayscale"));
    }

    #[test]
    fn empty_pixels_creates_empty_result_via_encode_to_jpeg() {
        // pixels_to_jpeg itself doesn't check for empty, but encode_to_jpeg does
        // Test the raw function with valid but minimal data
        let pixels = solid_color_pixels(1, 1, 0, 0, 0);
        let result = pixels_to_jpeg(&pixels, 1, 1, bridge::ColorFormat::RGB);
        assert!(result.is_ok());
    }

    #[test]
    fn invalid_buffer_size_fails() {
        // Buffer too small for dimensions
        let pixels = vec![0u8; 100]; // Not enough for 64x64x3
        let result = pixels_to_jpeg(&pixels, 64, 64, bridge::ColorFormat::RGB);

        assert!(result.is_err());
    }

    #[test]
    fn various_resolutions() {
        let test_cases = [
            (640, 480, "VGA"),
            (1280, 720, "HD"),
            (1920, 1080, "Full HD"),
        ];

        for (width, height, label) in test_cases {
            let pixels = solid_color_pixels(width, height, 128, 128, 128);
            let result = pixels_to_jpeg(&pixels, width, height, bridge::ColorFormat::RGB);

            assert!(result.is_ok(), "Failed for {}", label);
            let jpeg = result.unwrap();
            assert!(!jpeg.is_empty(), "Empty JPEG for {}", label);
            assert_eq!(
                &jpeg[0..2],
                &[0xFF, 0xD8],
                "Invalid JPEG header for {}",
                label
            );
        }
    }
}
