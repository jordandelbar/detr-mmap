use crate::config::Config;

pub fn setup_logging(config: &Config) {
    common::setup_logging(config.log_level.clone(), config.environment.clone());
}
