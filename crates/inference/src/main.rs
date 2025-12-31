use inference::{
    InferenceConfig, InferenceService,
    backend::{InferenceBackend, ort::OrtBackend},
    logging::setup_logging,
};

fn main() -> anyhow::Result<()> {
    let config = InferenceConfig::from_env()?;

    setup_logging(&config);

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
