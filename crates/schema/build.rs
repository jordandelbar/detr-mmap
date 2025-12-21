use std::path::Path;

fn main() {
    let frame_schema = Path::new("frame.fbs");
    let detection_schema = Path::new("detection.fbs");

    println!("cargo:rerun-if-changed={}", frame_schema.display());
    println!("cargo:rerun-if-changed={}", detection_schema.display());

    flatc_rust::run(flatc_rust::Args {
        inputs: &[frame_schema, detection_schema],
        out_dir: Path::new("src/"),
        ..Default::default()
    })
    .expect("Failed to generate Rust code from FlatBuffer schemas");
}
