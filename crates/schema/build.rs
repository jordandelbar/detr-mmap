use std::fs;
use std::path::Path;
use std::process::Command;

fn main() {
    let trace_context_schema = Path::new("trace_context.fbs");
    let frame_schema = Path::new("frame.fbs");
    let detection_schema = Path::new("detection.fbs");

    println!("cargo:rerun-if-changed={}", trace_context_schema.display());
    println!("cargo:rerun-if-changed={}", frame_schema.display());
    println!("cargo:rerun-if-changed={}", detection_schema.display());

    flatc_rust::run(flatc_rust::Args {
        inputs: &[trace_context_schema, frame_schema, detection_schema],
        out_dir: Path::new("src/"),
        ..Default::default()
    })
    .expect("Failed to generate Rust code from FlatBuffer schemas");

    let trace_context_path = Path::new("src/trace_context_generated.rs");
    let mut content =
        fs::read_to_string(trace_context_path).expect("Failed to trace_context_generated.rs");
    content.push_str("\npub use bridge::schema::*;\n");
    fs::write(trace_context_path, content).expect("Failed to write trace_context_generated.rs");

    let _ = Command::new("rustfmt")
        .args([
            "src/trace_context_generated.rs",
            "src/frame_generated.rs",
            "src/detection_generated.rs",
        ])
        .status();
}
