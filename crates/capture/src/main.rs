use bridge::SentryControl;
use capture::{camera::Camera, config::CameraConfig, logging::setup_logging};
use signal_hook::{
    consts::{SIGINT, SIGTERM},
    flag,
};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

fn main() -> anyhow::Result<()> {
    let config = CameraConfig::from_env()?;
    setup_logging(&config);
    let shutdown = Arc::new(AtomicBool::new(false));

    flag::register(SIGTERM, Arc::clone(&shutdown))?;
    flag::register(SIGINT, Arc::clone(&shutdown))?;

    tracing::info!("Signal handlers registered (SIGTERM, SIGINT)");

    let mut camera = Camera::build(config).expect("failed to build camera");

    let sentry_control = SentryControl::new("/dev/shm/bridge_sentry_control")
        .expect("failed to create sentry control");

    tracing::info!("Sentry control initialized in shared memory");

    match camera.run(&shutdown, &sentry_control) {
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
