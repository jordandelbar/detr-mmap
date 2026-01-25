use common::TelemetryGuard;
use inference::{InferenceBackend, InferenceConfig, InferenceService, logging::setup_logging};

#[cfg(all(feature = "ort-backend", not(feature = "trt-backend")))]
use inference::backend::ort::OrtBackend as Backend;

#[cfg(feature = "trt-backend")]
use inference::backend::trt::TrtBackend as Backend;

#[cfg(not(any(feature = "ort-backend", feature = "trt-backend")))]
compile_error!("At least one backend feature must be enabled: 'ort-backend' or 'trt-backend'");

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config = InferenceConfig::from_env()?;

    // TelemetryGuard initializes the tracing subscriber, so only call setup_logging if not using telemetry
    let _telemetry = if let Some(endpoint) = config.otel_endpoint.as_ref() {
        Some(TelemetryGuard::init("inference", endpoint, config.environment.clone())?)
    } else {
        setup_logging(&config);
        None
    };

    tracing::info!(
        config = ?config,
        "Loaded configuration"
    );

    tracing::info!("Loading inference model");
    let backend = Backend::load_model(&config.model_path)?;
    tracing::info!("Model loaded successfully");

    let service = InferenceService::new(backend, config);
    service.run()
}
