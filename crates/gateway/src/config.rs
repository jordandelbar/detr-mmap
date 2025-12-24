// Re-export from common crate for convenience
pub use common::{Environment, LogLevel};

#[derive(Debug, Clone)]
pub struct Config {
    pub log_level: LogLevel,
    pub environment: Environment,
}

impl Config {
    /// Load configuration from environment variables with sensible defaults
    pub fn from_env() -> Self {
        Self {
            log_level: LogLevel::from_env(),
            environment: Environment::from_env(),
        }
    }
}
