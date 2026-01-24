use common::{Environment, get_env, get_env_opt};
use preprocess::DEFAULT_INPUT_SIZE;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExecutionProvider {
    #[default]
    Cpu,
    Cuda,
}

impl ExecutionProvider {
    pub fn from_env() -> Self {
        match std::env::var("EXECUTION_PROVIDER").ok() {
            Some(s) if s.eq_ignore_ascii_case("cuda") || s.eq_ignore_ascii_case("gpu") => {
                Self::Cuda
            }
            _ => Self::Cpu,
        }
    }
}

#[derive(Debug, Clone)]
pub struct InferenceConfig {
    pub environment: Environment,
    pub model_path: String,
    pub input_size: (u32, u32),
    pub poll_interval_ms: u64,
    pub confidence_threshold: f32,
    pub otel_endpoint: Option<String>,
}

impl InferenceConfig {
    /// Load configuration from environment variables with sensible defaults
    pub fn from_env() -> anyhow::Result<Self> {
        Ok(Self {
            environment: Environment::from_env(),
            model_path: get_env(
                "MODEL_PATH",
                "/models/rfdetr_S/rfdetr_int8.engine".to_string(),
            ),
            input_size: (
                get_env("INPUT_WIDTH", DEFAULT_INPUT_SIZE.0),
                get_env("INPUT_HEIGHT", DEFAULT_INPUT_SIZE.1),
            ),
            poll_interval_ms: get_env("POLL_INTERVAL_MS", 100),
            confidence_threshold: get_env("CONFIDENCE_THRESHOLD", 0.7),
            otel_endpoint: get_env_opt("OTEL_EXPORTER_OTLP_ENDPOINT"),
        })
    }

    /// Create default configuration for testing
    #[cfg(test)]
    pub fn test_default() -> Self {
        Self {
            environment: Environment::Development,
            model_path: "/models/rfdetr.onnx".to_string(),
            input_size: DEFAULT_INPUT_SIZE,
            poll_interval_ms: 100,
            confidence_threshold: 0.7,
            otel_endpoint: None,
        }
    }
}
