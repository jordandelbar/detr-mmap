use crate::config::GatewayConfig;

pub fn setup_logging(config: &GatewayConfig) {
    common::setup_logging(config.environment.clone());
}
