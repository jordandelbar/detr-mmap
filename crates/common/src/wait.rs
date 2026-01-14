use std::time::Duration;

pub fn wait_for_resource<F, T, E>(mut connect: F, poll_interval_ms: u64, resource_name: &str) -> T
where
    F: FnMut() -> Result<T, E>,
    E: std::fmt::Display,
{
    loop {
        match connect() {
            Ok(resource) => {
                tracing::info!("{} connected", resource_name);
                return resource;
            }
            Err(e) => {
                tracing::debug!("Waiting for {} ({})", resource_name, e);
                std::thread::sleep(Duration::from_millis(poll_interval_ms));
            }
        }
    }
}

#[cfg(feature = "async")]
pub async fn wait_for_resource_async<F, T, E>(
    mut connect: F,
    poll_interval_ms: u64,
    resource_name: &str,
) -> T
where
    F: FnMut() -> Result<T, E>,
    E: std::fmt::Display,
{
    loop {
        match connect() {
            Ok(resource) => {
                tracing::info!("{} connected", resource_name);
                return resource;
            }
            Err(e) => {
                tracing::debug!("Waiting for {} ({})", resource_name, e);
                tokio::time::sleep(Duration::from_millis(poll_interval_ms)).await;
            }
        }
    }
}
