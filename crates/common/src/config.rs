use std::{env, str::FromStr};

/// Get an environment variable and parse it, returning a default if not set or parse fails.
pub fn get_env<T: FromStr>(key: &str, default: T) -> T {
    env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// Get an optional environment variable. Returns `None` if not set, `Some(T)` if set and parseable.
pub fn get_env_opt<T: FromStr>(key: &str) -> Option<T> {
    env::var(key).ok().and_then(|s| s.parse().ok())
}

#[derive(Debug, Clone)]
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

    pub fn from_env() -> Self {
        match env::var("ENVIRONMENT")
            .unwrap_or_else(|_| "development".to_string())
            .to_lowercase()
            .as_str()
        {
            "production" | "prod" => Environment::Production,
            _ => Environment::Development,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    #[serial]
    fn test_get_env_default() {
        let key = "TEST_GET_ENV_KEY";
        unsafe {
            std::env::remove_var(key);
        }

        assert_eq!(get_env(key, 42), 42);
    }

    #[test]
    #[serial]
    fn test_get_env_from_env() {
        let key = "TEST_GET_ENV_KEY";

        unsafe {
            std::env::set_var(key, "123");
        }

        assert_eq!(get_env(key, 42), 123);

        unsafe {
            std::env::remove_var(key);
        }
    }

    #[test]
    #[serial]
    fn test_get_env_opt() {
        let key = "TEST_GET_ENV_OPT_KEY";

        unsafe {
            std::env::remove_var(key);
        }
        assert_eq!(get_env_opt::<i32>(key), None);

        unsafe {
            std::env::set_var(key, "100");
        }
        assert_eq!(get_env_opt::<i32>(key), Some(100));

        unsafe {
            std::env::set_var(key, "invalid");
        }
        assert_eq!(get_env_opt::<i32>(key), None);

        unsafe {
            std::env::remove_var(key);
        }
    }

    #[test]
    fn test_environment_as_str() {
        assert_eq!(Environment::Development.as_str(), "development");
        assert_eq!(Environment::Production.as_str(), "production");
    }

    #[test]
    #[serial]
    fn test_environment_from_env() {
        let key = "ENVIRONMENT";
        let original = env::var(key);

        unsafe {
            env::remove_var(key);
        }
        assert!(matches!(Environment::from_env(), Environment::Development));

        // Test production variants
        unsafe {
            env::set_var(key, "production");
        }
        assert!(matches!(Environment::from_env(), Environment::Production));

        unsafe {
            env::set_var(key, "Production");
        }
        assert!(matches!(Environment::from_env(), Environment::Production));

        unsafe {
            env::set_var(key, "prod");
        }
        assert!(matches!(Environment::from_env(), Environment::Production));

        // Test development explicit
        unsafe {
            env::set_var(key, "development");
        }
        assert!(matches!(Environment::from_env(), Environment::Development));

        // Test fallback
        unsafe {
            env::set_var(key, "staging");
        }
        assert!(matches!(Environment::from_env(), Environment::Development));

        // Restore original
        match original {
            Ok(val) => unsafe { env::set_var(key, val) },
            Err(_) => unsafe { env::remove_var(key) },
        }
    }
}
