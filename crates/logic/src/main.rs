use axum::{
    Router,
    extract::{State, WebSocketUpgrade, ws::WebSocket},
    response::IntoResponse,
    routing::get,
};
use bridge::MmapReader;
use image::{ImageBuffer, RgbImage};
use serde::{Deserialize, Serialize};
use std::io::Cursor;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::broadcast;
use tokio::time;
use tower_http::cors::CorsLayer;

const FRAME_MMAP_PATH: &str = "/dev/shm/bridge_frame_buffer";
const DETECTION_MMAP_PATH: &str = "/dev/shm/bridge_detection_buffer";
const POLL_INTERVAL_MS: u64 = 16; // ~60fps
const WS_ADDR: &str = "0.0.0.0:8080";

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Detection {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    confidence: f32,
    class_id: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FrameMessage {
    frame_number: u64,
    timestamp_ns: u64,
    width: u32,
    height: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    detections: Option<Vec<Detection>>,
    status: String,
}

#[derive(Clone)]
struct FramePacket {
    metadata: FrameMessage,
    jpeg_data: Vec<u8>,
}

#[derive(Clone)]
struct AppState {
    tx: Arc<broadcast::Sender<FramePacket>>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("Logic service starting...");
    println!("Frame source: {}", FRAME_MMAP_PATH);
    println!("Detection source: {}", DETECTION_MMAP_PATH);
    println!("WebSocket server: ws://{}\n", WS_ADDR);

    let (tx, _) = broadcast::channel::<FramePacket>(10);
    let state = AppState { tx: Arc::new(tx) };

    let state_clone = state.clone();
    tokio::spawn(async move {
        if let Err(e) = poll_buffers(state_clone.tx).await {
            eprintln!("Buffer polling error: {}", e);
        }
    });

    let app = Router::new()
        .route("/ws", get(ws_handler))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(WS_ADDR).await?;
    println!("✓ WebSocket server listening on {}\n", WS_ADDR);

    axum::serve(listener, app).await?;

    Ok(())
}

async fn ws_handler(ws: WebSocketUpgrade, State(state): State<AppState>) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_socket(socket, state))
}

async fn handle_socket(mut socket: WebSocket, state: AppState) {
    println!("New WebSocket connection");

    let mut rx = state.tx.subscribe();

    while let Ok(packet) = rx.recv().await {
        let json = match serde_json::to_vec(&packet.metadata) {
            Ok(j) => j,
            Err(e) => {
                eprintln!("JSON serialization error: {}", e);
                continue;
            }
        };

        let mut binary_msg = Vec::with_capacity(4 + json.len() + packet.jpeg_data.len());
        binary_msg.extend_from_slice(&(json.len() as u32).to_le_bytes());
        binary_msg.extend_from_slice(&json);
        binary_msg.extend_from_slice(&packet.jpeg_data);

        if socket
            .send(axum::extract::ws::Message::Binary(binary_msg))
            .await
            .is_err()
        {
            println!("Client disconnected");
            break;
        }
    }
}

async fn poll_buffers(tx: Arc<broadcast::Sender<FramePacket>>) -> anyhow::Result<()> {
    let mut frame_reader = loop {
        match MmapReader::new(FRAME_MMAP_PATH) {
            Ok(reader) => {
                println!("✓ Frame buffer connected");
                break reader;
            }
            Err(_) => {
                println!("⧗ Waiting for frame buffer...");
                time::sleep(Duration::from_millis(500)).await;
            }
        }
    };

    let mut detection_reader = loop {
        match MmapReader::new(DETECTION_MMAP_PATH) {
            Ok(reader) => {
                println!("✓ Detection buffer connected");
                break reader;
            }
            Err(_) => {
                println!("⧗ Waiting for detection buffer...");
                time::sleep(Duration::from_millis(500)).await;
            }
        }
    };

    println!("Polling buffers at {}ms intervals\n", POLL_INTERVAL_MS);

    let mut interval = time::interval(Duration::from_millis(POLL_INTERVAL_MS));

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
                    eprintln!("Image encoding error: {}", e);
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
        println!(
            "Frame #{}: {} detections ({})",
            packet.metadata.frame_number, det_count, packet.metadata.status
        );

        let _ = tx.send(packet);
    }
}

fn pixels_to_jpeg(pixel_data: &[u8], width: u32, height: u32, format: schema::ColorFormat) -> anyhow::Result<Vec<u8>> {
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
            return Err(anyhow::anyhow!("Grayscale format not supported for JPEG encoding"));
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
