use gateway::{camera::run_camera, config::get_configuration, logging::setup_logging};

fn main() {
    let config = get_configuration().expect("failed to load configuration");
    setup_logging(config);
    let _ = run_camera();
}
