use common::{Environment, get_env, get_env_opt};

#[derive(Debug, Clone)]
pub struct GatewayConfig {
    pub environment: Environment,
    pub poll_interval_ms: u64,
    pub ws_addr: String,
    pub channel_capacity: usize,
    pub otel_endpoint: Option<String>,
}

impl GatewayConfig {
    pub fn from_env() -> Self {
        Self {
            environment: Environment::from_env(),
            poll_interval_ms: get_env("GATEWAY_POLL_INTERVAL_MS", 16), // ~60fps
            ws_addr: get_env("GATEWAY_WS_ADDR", "0.0.0.0:8080".to_string()),
            channel_capacity: get_env("GATEWAY_CHANNEL_CAPACITY", 10),
            otel_endpoint: get_env_opt("OTEL_EXPORTER_OTLP_ENDPOINT"),
        }
    }

    #[cfg(test)]
    pub fn test_default() -> Self {
        Self {
            environment: Environment::Development,
            poll_interval_ms: 16,
            ws_addr: "0.0.0.0:8080".to_string(),
            channel_capacity: 10,
            otel_endpoint: None,
        }
    }
}
