use std::time::Duration;

/// Retry a function with exponential backoff
///
/// # Arguments
/// * `f` - The function to retry
/// * `max_retries` - Maximum number of retry attempts
/// * `base_delay_ms` - Initial delay in milliseconds (doubles each retry)
/// * `operation_name` - Human-readable name for logging
pub fn retry_with_backoff<F, T, E>(
    mut f: F,
    max_retries: u32,
    base_delay_ms: u64,
    operation_name: &str,
) -> Result<T, E>
where
    F: FnMut() -> Result<T, E>,
    E: std::fmt::Display,
{
    for attempt in 0..max_retries {
        match f() {
            Ok(result) => return Ok(result),
            Err(e) => {
                if attempt < max_retries - 1 {
                    let delay_ms = base_delay_ms * 2_u64.pow(attempt);
                    tracing::warn!(
                        "{} failed (attempt {}/{}): {}. Retrying in {}ms...",
                        operation_name,
                        attempt + 1,
                        max_retries,
                        e,
                        delay_ms
                    );
                    std::thread::sleep(Duration::from_millis(delay_ms));
                } else {
                    tracing::error!(
                        "{} failed after {} attempts: {}",
                        operation_name,
                        max_retries,
                        e
                    );
                    return Err(e);
                }
            }
        }
    }
    unreachable!()
}
