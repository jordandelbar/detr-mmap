use capture::{camera::Camera, config::CameraConfig, logging::setup_logging};
use signal_hook::consts::{SIGINT, SIGTERM};
use signal_hook::flag;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

fn main() -> anyhow::Result<()> {
    let config = CameraConfig::from_env()?;
    setup_logging(&config);

    // Set up graceful shutdown on SIGTERM (Kubernetes) and SIGINT (Ctrl+C)
    let shutdown = Arc::new(AtomicBool::new(false));

    flag::register(SIGTERM, Arc::clone(&shutdown))?;
    flag::register(SIGINT, Arc::clone(&shutdown))?;

    tracing::info!("Signal handlers registered (SIGTERM, SIGINT)");

    let mut camera = Camera::build(config).expect("failed to build camera");

    match camera.run(&shutdown) {
        Ok(_) => {
            tracing::info!("Camera capture stopped gracefully");
            Ok(())
        }
        Err(e) => {
            tracing::error!("Camera capture failed: {}", e);
            anyhow::bail!("Camera capture error: {}", e)
        }
    }
}
