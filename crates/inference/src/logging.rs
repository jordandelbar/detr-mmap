use crate::config::InferenceConfig;

pub fn setup_logging(config: &InferenceConfig) {
    common::setup_logging(config.environment.clone());
}
