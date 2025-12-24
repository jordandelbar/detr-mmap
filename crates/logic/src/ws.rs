use crate::config::LogicConfig;
use crate::state::AppState;
use axum::{
    extract::{ws::WebSocket, State, WebSocketUpgrade},
    response::IntoResponse,
    routing::get,
    Router,
};
use tower_http::cors::CorsLayer;

pub async fn run_server(config: LogicConfig, state: AppState) -> anyhow::Result<()> {
    let app = Router::new()
        .route("/ws", get(ws_handler))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(&config.ws_addr).await?;
    tracing::info!("WebSocket server listening on {}", config.ws_addr);

    axum::serve(listener, app).await?;

    Ok(())
}

async fn ws_handler(ws: WebSocketUpgrade, State(state): State<AppState>) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_socket(socket, state))
}

async fn handle_socket(mut socket: WebSocket, state: AppState) {
    tracing::info!("New WebSocket connection established");

    let mut rx = state.tx.subscribe();

    while let Ok(packet) = rx.recv().await {
        let json = match serde_json::to_vec(&packet.metadata) {
            Ok(j) => j,
            Err(e) => {
                tracing::error!("JSON serialization error: {}", e);
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
            tracing::info!("WebSocket client disconnected");
            break;
        }
    }
}
