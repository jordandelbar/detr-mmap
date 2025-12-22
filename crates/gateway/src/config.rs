use serde::Deserialize;

#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LogLevel {
    Debug,
    Info,
}

impl LogLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            LogLevel::Debug => "debug",
            LogLevel::Info => "info",
        }
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Environment {
    Development,
    Production,
}

impl Environment {
    pub fn as_str(&self) -> &'static str {
        match self {
            Environment::Development => "development",
            Environment::Production => "production",
        }
    }
}

impl TryFrom<String> for Environment {
    type Error = String;

    fn try_from(s: String) -> Result<Self, Self::Error> {
        match s.to_lowercase().as_str() {
            "development" => Ok(Self::Development),
            "production" => Ok(Self::Production),
            other => Err(format!(
                "{} is not a supported environment. Use either `development` or `production`.",
                other
            )),
        }
    }
}

#[derive(Deserialize)]
pub struct Config {
    pub log_level: LogLevel,
    pub environment: Environment,
}

pub fn get_configuration() -> Result<Config, config::ConfigError> {
    let config = config::Config::builder()
        .set_default("log_level", "info")?
        .set_default("environment", "development")?
        .add_source(
            config::Environment::with_prefix("GATEWAY")
                .prefix_separator("_")
                .separator("__")
                .try_parsing(true),
        )
        .build()?;

    let config: Config = config.try_deserialize::<Config>()?;

    Ok(config)
}
