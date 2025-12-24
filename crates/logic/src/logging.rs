use crate::config::LogicConfig;

pub fn setup_logging(config: &LogicConfig) {
    common::setup_logging(config.environment.clone());
}
