use std::time::Duration;

/// Configuration for retry behavior with exponential backoff
///
/// Default values are optimized for low-latency pipelines:
/// - 20 attempts with 100µs base delay
/// - Exponential backoff capped at 2ms
/// - Total worst-case wait: ~7ms
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of retry attempts before returning NoDataAvailable
    pub max_attempts: u32,
    /// Initial delay between retries (doubles each attempt)
    pub base_delay: Duration,
    /// Maximum delay cap (backoff won't exceed this)
    pub max_delay: Duration,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 20,
            base_delay: Duration::from_micros(100),
            max_delay: Duration::from_millis(2),
        }
    }
}

impl RetryConfig {
    /// Calculate delay for a given attempt using exponential backoff
    pub(crate) fn delay_for_attempt(&self, attempt: u32) -> Duration {
        self.base_delay
            .saturating_mul(2u32.pow(attempt))
            .min(self.max_delay)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = RetryConfig::default();
        assert_eq!(config.max_attempts, 20);
        assert_eq!(config.base_delay, Duration::from_micros(100));
        assert_eq!(config.max_delay, Duration::from_millis(2));
    }

    #[test]
    fn test_exponential_backoff_calculation() {
        let config = RetryConfig::default();

        // 100µs * 2^0 = 100µs
        assert_eq!(config.delay_for_attempt(0), Duration::from_micros(100));
        // 100µs * 2^1 = 200µs
        assert_eq!(config.delay_for_attempt(1), Duration::from_micros(200));
        // 100µs * 2^2 = 400µs
        assert_eq!(config.delay_for_attempt(2), Duration::from_micros(400));
        // 100µs * 2^3 = 800µs
        assert_eq!(config.delay_for_attempt(3), Duration::from_micros(800));
        // 100µs * 2^4 = 1600µs
        assert_eq!(config.delay_for_attempt(4), Duration::from_micros(1600));
        // 100µs * 2^5 = 3200µs, but capped at 2000µs
        assert_eq!(config.delay_for_attempt(5), Duration::from_millis(2));
        // Higher attempts stay capped
        assert_eq!(config.delay_for_attempt(10), Duration::from_millis(2));
    }

    #[test]
    fn test_custom_config() {
        let config = RetryConfig {
            max_attempts: 5,
            base_delay: Duration::from_micros(50),
            max_delay: Duration::from_micros(500),
        };

        assert_eq!(config.delay_for_attempt(0), Duration::from_micros(50));
        assert_eq!(config.delay_for_attempt(1), Duration::from_micros(100));
        assert_eq!(config.delay_for_attempt(2), Duration::from_micros(200));
        assert_eq!(config.delay_for_attempt(3), Duration::from_micros(400));
        // Capped at 500µs
        assert_eq!(config.delay_for_attempt(4), Duration::from_micros(500));
    }
}
