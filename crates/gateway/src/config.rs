// Re-export from common crate for convenience
pub use common::Environment;

#[derive(Debug, Clone)]
pub struct Config {
    pub environment: Environment,
}

impl Config {
    /// Load configuration from environment variables with sensible defaults
    pub fn from_env() -> Self {
        Self {
            environment: Environment::from_env(),
        }
    }
}
