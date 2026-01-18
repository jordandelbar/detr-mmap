use common::TelemetryGuard;
use inference::{
    InferenceConfig, InferenceService, backend::InferenceBackend, logging::setup_logging,
};

#[cfg(all(feature = "ort-backend", not(feature = "trt-backend")))]
use inference::backend::ort::OrtBackend as Backend;

#[cfg(feature = "trt-backend")]
use inference::backend::trt::TrtBackend as Backend;

#[cfg(not(any(feature = "ort-backend", feature = "trt-backend")))]
compile_error!("At least one backend feature must be enabled: 'ort-backend' or 'trt-backend'");

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config = InferenceConfig::from_env()?;

    let _telemetry = config
        .otel_endpoint
        .as_ref()
        .map(|endpoint| TelemetryGuard::init("inference", endpoint))
        .transpose()?;

    setup_logging(&config);

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
