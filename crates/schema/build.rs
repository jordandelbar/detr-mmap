use std::path::Path;

fn main() {
    let schema_path = Path::new("frame.fbs");

    println!("cargo:rerun-if-changed={}", schema_path.display());

    flatc_rust::run(flatc_rust::Args {
        inputs: &[schema_path],
        out_dir: Path::new("src/"),
        ..Default::default()
    })
    .expect("Failed to generate Rust code from FlatBuffer schema");
}
