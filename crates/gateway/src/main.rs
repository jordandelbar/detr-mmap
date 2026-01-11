use gateway::{
    config::GatewayConfig, logging::setup_logging, polling::BufferPoller, state::AppState, ws,
};
use std::sync::Arc;
use tokio::sync::broadcast;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config = GatewayConfig::from_env();

    setup_logging(&config);

    tracing::info!("Gateway service starting");
    tracing::info!("WebSocket endpoint: ws://{}/ws", config.ws_addr);

    let (tx, _rx) = broadcast::channel(config.channel_capacity);
    let state = AppState { tx: Arc::new(tx) };
    let poll_tx = state.tx.clone();

    tokio::spawn(async move {
        match BufferPoller::build(poll_tx).await {
            Ok(poller) => {
                if let Err(e) = poller.run().await {
                    tracing::error!("Buffer polling error: {}", e);
                }
            }
            Err(e) => {
                tracing::error!("Failed to build buffer poller: {}", e);
            }
        }
    });

    ws::run_server(config, state).await?;

    Ok(())
}
