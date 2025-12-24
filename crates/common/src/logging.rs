use crate::config::{Environment, LogLevel};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

/// Initialize tracing subscriber with pretty formatting for development
/// and JSON formatting for production
pub fn setup_logging(log_level: LogLevel, environment: Environment) {
    let log_level_str = log_level.as_str();
    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| log_level_str.into());
    let registry = tracing_subscriber::registry().with(env_filter);

    match environment {
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
}
