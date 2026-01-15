use common::{Environment, get_env};

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
        Ok(Self {
            environment: Environment::from_env(),
            model_path: get_env(
                "MODEL_PATH",
                "/home/jdelbar/Documents/projects/bridge-rt/models/model_fp16.engine".to_string(),
            ),
            input_size: (get_env("INPUT_WIDTH", 640), get_env("INPUT_HEIGHT", 640)),
            poll_interval_ms: get_env("POLL_INTERVAL_MS", 100),
            confidence_threshold: get_env("CONFIDENCE_THRESHOLD", 0.5),
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
