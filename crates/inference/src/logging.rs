use crate::config::InferenceConfig;

pub fn setup_logging(config: &InferenceConfig) {
    common::setup_logging(config.log_level.clone(), config.environment.clone());
}
