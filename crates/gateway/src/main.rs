use gateway::{camera::Camera, config::CameraConfig, logging::setup_logging};

fn main() -> anyhow::Result<()> {
    let config = CameraConfig::from_env()?;
    setup_logging(&config);
    let mut camera = Camera::build(config).expect("failed to build camera");
    let _ = camera.run();
    Ok(())
}
