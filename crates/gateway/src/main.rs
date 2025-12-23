use gateway::{
    camera::{Camera, CameraConfig},
    config::get_configuration,
    logging::setup_logging,
};

fn main() {
    let config = get_configuration().expect("failed to load configuration");
    setup_logging(config);

    let camera_config = CameraConfig {
        camera_id: 0,
        device_id: 0,
        mmap_path: "/dev/shm/bridge_frame_buffer".to_string(),
        mmap_size: 8 * 1024 * 1024,
    };

    let mut camera = Camera::build(camera_config).expect("failed to build camera");
    let _ = camera.run();
}
