use axum::{
    extract::{ws::WebSocket, State, WebSocketUpgrade},
    response::IntoResponse,
    routing::get,
    Router,
};
use base64::{engine::general_purpose, Engine as _};
use bridge::MmapReader;
use serde::{Deserialize, Serialize};
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
    image_base64: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    detections: Option<Vec<Detection>>,
    status: String,
}

#[derive(Clone)]
struct AppState {
    tx: Arc<broadcast::Sender<FrameMessage>>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("Logic service starting...");
    println!("Frame source: {}", FRAME_MMAP_PATH);
    println!("Detection source: {}", DETECTION_MMAP_PATH);
    println!("WebSocket server: ws://{}\n", WS_ADDR);

    let (tx, _) = broadcast::channel::<FrameMessage>(10);
    let state = AppState {
        tx: Arc::new(tx),
    };

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

    while let Ok(msg) = rx.recv().await {
        let json = match serde_json::to_string(&msg) {
            Ok(j) => j,
            Err(e) => {
                eprintln!("JSON serialization error: {}", e);
                continue;
            }
        };

        if socket.send(axum::extract::ws::Message::Text(json)).await.is_err() {
            println!("Client disconnected");
            break;
        }
    }
}

async fn poll_buffers(tx: Arc<broadcast::Sender<FrameMessage>>) -> anyhow::Result<()> {
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

        let has_frame = frame_reader.has_new_data();
        let has_detection = detection_reader.has_new_data();

        let msg = match (has_frame, has_detection) {
            (false, false) => continue,
            (true, false) => {
                let frame = flatbuffers::root::<schema::Frame>(frame_reader.buffer())?;
                let pixels = frame.pixels().map(|p| p.bytes()).unwrap_or(&[]);

                FrameMessage {
                    frame_number: frame.frame_number(),
                    timestamp_ns: frame.timestamp_ns(),
                    width: frame.width(),
                    height: frame.height(),
                    image_base64: Some(general_purpose::STANDARD.encode(pixels)),
                    detections: None,
                    status: "frame_only".to_string(),
                }
            }
            (false, true) => {
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

                FrameMessage {
                    frame_number: detection.frame_number(),
                    timestamp_ns: detection.timestamp_ns(),
                    width: 0,
                    height: 0,
                    image_base64: None,
                    detections: dets,
                    status: "detection_only".to_string(),
                }
            }
            (true, true) => {
                let frame = flatbuffers::root::<schema::Frame>(frame_reader.buffer())?;
                let detection =
                    flatbuffers::root::<schema::DetectionResult>(detection_reader.buffer())?;

                let pixels = frame.pixels().map(|p| p.bytes()).unwrap_or(&[]);
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

                FrameMessage {
                    frame_number: frame.frame_number(),
                    timestamp_ns: frame.timestamp_ns(),
                    width: frame.width(),
                    height: frame.height(),
                    image_base64: Some(general_purpose::STANDARD.encode(pixels)),
                    detections: dets,
                    status: "complete".to_string(),
                }
            }
        };

        if has_frame {
            frame_reader.mark_read();
        }
        if has_detection {
            detection_reader.mark_read();
        }

        println!(
            "Broadcasting frame #{} - status: {}",
            msg.frame_number, msg.status
        );

        let _ = tx.send(msg);
    }
}
