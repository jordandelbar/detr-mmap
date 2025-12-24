use inference::{
    backend::{ort::OrtBackend, InferenceBackend},
    logging::setup_logging,
    InferenceConfig, InferenceService,
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
