use anyhow::Context;
use bridge::SentryControl;
use capture::{camera::Camera, config::CameraConfig, logging::setup_logging};
use common::TelemetryGuard;
use signal_hook::{
    consts::{SIGINT, SIGTERM},
    flag,
};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

fn main() -> anyhow::Result<()> {
    let config = CameraConfig::from_env()?;

    // TelemetryGuard requires a Tokio runtime for async OTLP exporters.
    // We must keep the runtime alive for the batch exporter to work.
    // It also initializes the tracing subscriber, so only call setup_logging if not using telemetry.
    let (_telemetry, _runtime) = if let Some(endpoint) = config.otel_endpoint.as_ref() {
        let rt = tokio::runtime::Runtime::new().expect("Failed to create Tokio runtime");
        let guard = rt.block_on(async { TelemetryGuard::init("capture", endpoint, config.environment.clone()) })?;
        (Some(guard), Some(rt))
    } else {
        setup_logging(&config);
        (None, None)
    };
    let shutdown = Arc::new(AtomicBool::new(false));

    flag::register(SIGTERM, Arc::clone(&shutdown))?;
    flag::register(SIGINT, Arc::clone(&shutdown))?;

    tracing::info!("Signal handlers registered (SIGTERM, SIGINT)");

    let mut camera = Camera::build(config)
        .context("Failed to initialize camera - check V4L2 device availability")?;

    let sentry_control = SentryControl::build()
        .context("Failed to create sentry control in shared memory (/dev/shm)")?;

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
