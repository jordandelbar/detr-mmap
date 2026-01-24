use preprocess::DEFAULT_INPUT_SIZE;
use preprocess::PreProcessor;
use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let mut pre = PreProcessor::new(DEFAULT_INPUT_SIZE);

    // Paths relative to workspace root
    let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();

    let input_dir = workspace_root.join("scripts/quantization/calibration_data");
    let output_dir = workspace_root.join("scripts/quantization/calibration_tensors");

    // Create output directory if it doesn't exist
    fs::create_dir_all(&output_dir)?;

    let mut idx = 0;

    for img_path in glob::glob(input_dir.join("*.jpg").to_str().unwrap())? {
        let img_path = img_path?;
        let img = image::open(&img_path)?.to_rgb8();
        let (tensor, _, _, _) = pre.preprocess_from_u8_slice(&img, img.width(), img.height())?;

        let output_path = output_dir.join(format!("calib_{idx:04}.bin"));
        dump_tensor(&output_path, &tensor)?;
        idx += 1;
        if idx >= 100 {
            break;
        }
    }

    println!("Processed {idx} images, tensors saved to {}", output_dir.display());

    Ok(())
}

fn dump_tensor(path: &std::path::Path, tensor: &ndarray::Array<f32, ndarray::IxDyn>) -> anyhow::Result<()> {
    let mut f = File::create(path)?;
    let slice = tensor.as_slice().unwrap();
    let bytes = bytemuck::cast_slice(slice);
    f.write_all(bytes)?;
    Ok(())
}
