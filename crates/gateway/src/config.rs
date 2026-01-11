use std::env;

pub use common::Environment;

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
        let environment = Environment::from_env();

        let poll_interval_ms = env::var("GATEWAY_POLL_INTERVAL_MS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(16); // ~60fps

        let ws_addr = env::var("GATEWAY_WS_ADDR").unwrap_or_else(|_| "0.0.0.0:8080".to_string());

        let channel_capacity = env::var("GATEWAY_CHANNEL_CAPACITY")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(10);

        Self {
            environment,
            poll_interval_ms,
            ws_addr,
            channel_capacity,
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
