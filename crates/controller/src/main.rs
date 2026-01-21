mod config;
mod mqtt_notifier;
mod service;
mod state_machine;

use common::TelemetryGuard;
use config::ControllerConfig;
use service::ControllerService;

fn main() -> anyhow::Result<()> {
    let config = ControllerConfig::from_env()?;

    // TelemetryGuard requires a Tokio runtime for async OTLP exporters.
    // We must keep the runtime alive for the batch exporter to work.
    // It also initializes the tracing subscriber, so only init fmt subscriber if not using telemetry.
    let (_telemetry, _runtime) = if let Some(endpoint) = config.otel_endpoint.as_ref() {
        let rt = tokio::runtime::Runtime::new().expect("Failed to create Tokio runtime");
        let guard = rt.block_on(async { TelemetryGuard::init("controller", endpoint) })?;
        (Some(guard), Some(rt))
    } else {
        tracing_subscriber::fmt()
            .with_target(false)
            .with_env_filter(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
            )
            .init();
        (None, None)
    };

    tracing::info!("Controller starting with config: {:?}", config);

    let service = ControllerService::new(config)?;
    service.run()
}
