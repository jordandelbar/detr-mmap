use std::env;

pub use common::Environment;

#[derive(Debug, Clone)]
pub struct InferenceConfig {
    pub environment: Environment,
    pub model_path: String,
    pub input_size: (u32, u32),
    pub poll_interval_ms: u64,
    pub confidence_threshold: f32,
}

impl InferenceConfig {
    /// Load configuration from environment variables with sensible defaults
    pub fn from_env() -> anyhow::Result<Self> {
        let environment = Environment::from_env();

        let model_path = env::var("MODEL_PATH").unwrap_or_else(|_| {
            "/home/jdelbar/Documents/projects/bridge-rt/models/model_fp16.engine".to_string()
        });

        let input_width = env::var("INPUT_WIDTH")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(640);

        let input_height = env::var("INPUT_HEIGHT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(640);

        let poll_interval_ms = env::var("POLL_INTERVAL_MS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(100);

        let confidence_threshold = env::var("CONFIDENCE_THRESHOLD")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.5);

        Ok(Self {
            environment,
            model_path,
            input_size: (input_width, input_height),
            poll_interval_ms,
            confidence_threshold,
        })
    }

    /// Create default configuration for testing
    #[cfg(test)]
    pub fn test_default() -> Self {
        Self {
            environment: Environment::Development,
            model_path: "/models/model.onnx".to_string(),
            input_size: (640, 640),
            poll_interval_ms: 100,
            confidence_threshold: 0.5,
        }
    }
}
