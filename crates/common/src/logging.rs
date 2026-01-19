use crate::config::Environment;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

/// Initialize tracing subscriber with pretty formatting for development
/// and JSON formatting for production.
///
/// Uses RUST_LOG environment variable for filtering (defaults to "info" if not set).
///
/// Also adds an OpenTelemetry layer that exports traces if a global tracer provider
/// has been initialized (e.g. via common::telemetry::init_telemetry).
pub fn setup_logging(environment: Environment) {
    let env_filter =
        tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into());

    let otel_layer = tracing_opentelemetry::layer();

    let registry = tracing_subscriber::registry()
        .with(env_filter)
        .with(otel_layer);

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
