use crate::config::Config;

pub fn setup_logging(config: &Config) {
    common::setup_logging(config.environment.clone());
}
