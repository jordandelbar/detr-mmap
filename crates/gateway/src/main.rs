use gateway::{
    camera::run_camera,
    config::{Environment, get_configuration},
};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

fn main() {
    let config = get_configuration().expect("failed to load configuration");
    let log_level = config.log_level.as_str();
    let env_filter =
        tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| log_level.into());
    let registry = tracing_subscriber::registry().with(env_filter);

    match config.environment {
        Environment::Production => {
            registry
                .with(tracing_subscriber::fmt::layer().json().with_level(true))
                .init();
        }
        Environment::Development => {
            registry
                .with(tracing_subscriber::fmt::layer().pretty().with_ansi(true))
                .init();
        }
    }

    let _ = run_camera();
}
