use logic::{config::LogicConfig, logging::setup_logging, polling, state::AppState, ws};
use std::sync::Arc;
use tokio::sync::broadcast;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config = LogicConfig::from_env();

    setup_logging(&config);

    tracing::info!("Logic service starting");
    tracing::info!("Frame source: {}", config.frame_mmap_path);
    tracing::info!("Detection source: {}", config.detection_mmap_path);
    tracing::info!("WebSocket endpoint: ws://{}/ws", config.ws_addr);

    let (tx, _rx) = broadcast::channel(config.channel_capacity);
    let state = AppState { tx: Arc::new(tx) };

    let poll_config = config.clone();
    let poll_tx = state.tx.clone();
    tokio::spawn(async move {
        if let Err(e) = polling::poll_buffers(poll_config, poll_tx).await {
            tracing::error!("Buffer polling error: {}", e);
        }
    });

    ws::run_server(config, state).await?;

    Ok(())
}
