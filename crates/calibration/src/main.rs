use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use preprocess::{DEFAULT_INPUT_SIZE, PreProcessor};
use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;

/// Generate calibration tensors for INT8 quantization.
///
/// This tool preprocesses JPEG images using the same pipeline as inference
/// and saves the resulting tensors as binary files for TensorRT calibration.
#[derive(Parser)]
#[command(version, about)]
struct Args {
    /// Directory containing input JPEG images
    #[arg(long, default_value = "scripts/quantization/calibration_data")]
    input_dir: PathBuf,

    /// Directory to save output tensors
    #[arg(long, default_value = "scripts/quantization/calibration_tensors")]
    output_dir: PathBuf,

    /// Number of images to process (0 = all)
    #[arg(long, default_value = "100")]
    count: usize,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let mut pre = PreProcessor::new(DEFAULT_INPUT_SIZE);

    // Resolve paths relative to workspace root if not absolute
    let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .ok_or_else(|| anyhow::anyhow!("Failed to find workspace root"))?
        .to_path_buf();

    let input_dir = if args.input_dir.is_absolute() {
        args.input_dir
    } else {
        workspace_root.join(&args.input_dir)
    };

    let output_dir = if args.output_dir.is_absolute() {
        args.output_dir
    } else {
        workspace_root.join(&args.output_dir)
    };

    // Create output directory if it doesn't exist
    fs::create_dir_all(&output_dir)?;

    // Collect all JPEG files
    let glob_pattern = input_dir
        .join("*.jpg")
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("Invalid path encoding"))?
        .to_owned();

    let image_paths: Vec<_> = glob::glob(&glob_pattern)?.filter_map(|p| p.ok()).collect();

    if image_paths.is_empty() {
        anyhow::bail!("No JPEG images found in {}", input_dir.display());
    }

    let total = if args.count == 0 {
        image_paths.len()
    } else {
        args.count.min(image_paths.len())
    };

    println!("Processing {} images from {}", total, input_dir.display());
    println!("Output directory: {}", output_dir.display());

    // Setup progress bar
    let pb = ProgressBar::new(total as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} ({eta})")?
            .progress_chars("#>-"),
    );

    let mut processed = 0;

    for img_path in image_paths.into_iter().take(total) {
        let img = image::open(&img_path)?.to_rgb8();
        let (tensor, _, _, _) = pre.preprocess_from_u8_slice(&img, img.width(), img.height())?;

        let output_path = output_dir.join(format!("calib_{processed:04}.bin"));
        dump_tensor(&output_path, &tensor)?;

        processed += 1;
        pb.inc(1);
    }

    pb.finish_with_message("done");

    println!(
        "\nProcessed {} images, tensors saved to {}",
        processed,
        output_dir.display()
    );

    Ok(())
}

fn dump_tensor(
    path: &std::path::Path,
    tensor: &ndarray::Array<f32, ndarray::IxDyn>,
) -> anyhow::Result<()> {
    let mut f = File::create(path)?;
    let slice = tensor
        .as_slice()
        .ok_or_else(|| anyhow::anyhow!("Tensor is not contiguous in memory"))?;
    let bytes = bytemuck::cast_slice(slice);
    f.write_all(bytes)?;
    Ok(())
}
