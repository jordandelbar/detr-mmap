mod config;
mod detection_reader;
mod mqtt_notifier;
mod service;
mod state_machine;

use config::ControllerConfig;
use service::ControllerService;

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_target(false)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let config = ControllerConfig::from_env()?;
    tracing::info!("Controller starting with config: {:?}", config);

    let service = ControllerService::new(config)?;
    service.run()
}
