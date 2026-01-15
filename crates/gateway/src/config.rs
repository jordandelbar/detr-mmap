use common::{Environment, get_env};

#[derive(Debug, Clone)]
pub struct GatewayConfig {
    pub environment: Environment,
    pub poll_interval_ms: u64,
    pub ws_addr: String,
    pub channel_capacity: usize,
}

impl GatewayConfig {
    /// Load configuration from environment variables with sensible defaults
    pub fn from_env() -> Self {
        Self {
            environment: Environment::from_env(),
            poll_interval_ms: get_env("GATEWAY_POLL_INTERVAL_MS", 16), // ~60fps
            ws_addr: get_env("GATEWAY_WS_ADDR", "0.0.0.0:8080".to_string()),
            channel_capacity: get_env("GATEWAY_CHANNEL_CAPACITY", 10),
        }
    }

    /// Create default configuration for testing
    #[cfg(test)]
    pub fn test_default() -> Self {
        Self {
            environment: Environment::Development,
            poll_interval_ms: 16,
            ws_addr: "0.0.0.0:8080".to_string(),
            channel_capacity: 10,
        }
    }
}
