use inference::{
    InferenceConfig, InferenceService,
    backend::{InferenceBackend, ort::OrtBackend},
};

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let config = InferenceConfig::from_env()?;

    tracing::info!(
        config = ?config,
        "Loaded configuration"
    );

    tracing::info!("Loading inference model");
    let backend = OrtBackend::load_model(&config.model_path)?;
    tracing::info!("Model loaded successfully");

    let service = InferenceService::new(backend, config);
    service.run()
}
