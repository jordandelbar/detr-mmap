#!/usr/bin/env python3
"""Prepare and publish RF-DETR INT8 model to HuggingFace.

This is the main orchestration script for maintainers. It:
1. Clones rf-detr if not exists
2. Exports the ONNX model (512x512)
3. Downloads calibration images
4. Generates calibration tensors (via Rust crate)
5. Builds calibration cache
6. Pushes ONNX + cache to HuggingFace
"""
import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
WORKSPACE_ROOT = SCRIPT_DIR.parent.parent


def run_step(name: str, cmd: list, dry_run: bool = False) -> None:
    """Run a pipeline step with logging."""
    print(f"\n{'='*60}")
    print(f"STEP: {name}")
    print(f"{'='*60}")

    if dry_run:
        print(f"[DRY RUN] Would run: {' '.join(str(c) for c in cmd)}")
        return

    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare and publish RF-DETR INT8 model to HuggingFace"
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        required=True,
        help="HuggingFace repository name (e.g., 'your-org/rfdetr-small-int8')",
    )
    parser.add_argument(
        "--model-variant",
        choices=["small", "base"],
        default="small",
        help="Model variant to export (default: small)",
    )
    parser.add_argument(
        "--calibration-images",
        type=int,
        default=100,
        help="Number of calibration images to use (default: 100)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip ONNX export (use existing model)",
    )
    parser.add_argument(
        "--skip-calibration-download",
        action="store_true",
        help="Skip downloading calibration images",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Skip uploading to HuggingFace",
    )
    args = parser.parse_args()

    # Paths
    model_dir = SCRIPT_DIR / "models" / f"rfdetr_{args.model_variant}"
    onnx_path = model_dir / "inference_model.onnx"
    cache_path = model_dir / "calibration.cache"
    calibration_data_dir = SCRIPT_DIR / "calibration_data"
    calibration_tensor_dir = SCRIPT_DIR / "calibration_tensors"

    print(f"Model directory: {model_dir}")
    print(f"HuggingFace repo: {args.hf_repo}")
    print(f"Dry run: {args.dry_run}")

    # Step 1: Export ONNX
    if not args.skip_export:
        run_step(
            "Export ONNX model",
            [
                sys.executable,
                str(SCRIPT_DIR / "export_onnx.py"),
                "--output-dir", str(model_dir),
                "--model-variant", args.model_variant,
            ],
            args.dry_run,
        )
    else:
        print(f"\nSkipping ONNX export (using existing: {onnx_path})")

    # Step 2: Download calibration images
    if not args.skip_calibration_download:
        run_step(
            "Download calibration images",
            [
                sys.executable,
                str(SCRIPT_DIR / "download_calibration_data.py"),
                "--output-dir", str(calibration_data_dir),
                "--count", str(args.calibration_images),
            ],
            args.dry_run,
        )
    else:
        print(f"\nSkipping calibration download (using existing: {calibration_data_dir})")

    # Step 3: Generate calibration tensors
    run_step(
        "Generate calibration tensors",
        [
            "cargo", "run", "-p", "calibration", "--release", "--",
            "--input-dir", str(calibration_data_dir),
            "--output-dir", str(calibration_tensor_dir),
            "--count", str(args.calibration_images),
        ],
        args.dry_run,
    )

    # Step 4: Build calibration cache
    run_step(
        "Build calibration cache",
        [
            sys.executable,
            str(SCRIPT_DIR / "build_calibration_cache.py"),
            "--onnx-path", str(onnx_path),
            "--cache-path", str(cache_path),
            "--calibration-dir", str(calibration_tensor_dir),
        ],
        args.dry_run,
    )

    # Step 5: Upload to HuggingFace
    if not args.skip_upload:
        print(f"\n{'='*60}")
        print("STEP: Upload to HuggingFace")
        print(f"{'='*60}")

        if args.dry_run:
            print(f"[DRY RUN] Would upload to {args.hf_repo}:")
            print(f"  - {onnx_path}")
            print(f"  - {cache_path}")
        else:
            from huggingface_hub import HfApi

            api = HfApi()

            # Create repo if it doesn't exist
            api.create_repo(args.hf_repo, exist_ok=True, repo_type="model")

            # Upload files
            print(f"Uploading {onnx_path.name}...")
            api.upload_file(
                path_or_fileobj=str(onnx_path),
                path_in_repo="inference_model.onnx",
                repo_id=args.hf_repo,
            )

            print(f"Uploading {cache_path.name}...")
            api.upload_file(
                path_or_fileobj=str(cache_path),
                path_in_repo="calibration.cache",
                repo_id=args.hf_repo,
            )

            # Create README for the HF repo
            readme_content = f"""# RF-DETR {args.model_variant.capitalize()} INT8

This repository contains the ONNX model and INT8 calibration cache for RF-DETR {args.model_variant.capitalize()}.

## Files

- `inference_model.onnx` - F32 ONNX model (512x512 input resolution)
- `calibration.cache` - TensorRT INT8 calibration cache

## Usage

### Quick Start

Download and build a TensorRT INT8 engine for your GPU:

```bash
# Install dependencies
uv pip install huggingface_hub tensorrt pycuda numpy

# Download the user build script
wget https://raw.githubusercontent.com/your-org/detr-mmap/main/scripts/model-pipeline/user_build_engine.py

# Build engine for your GPU
python user_build_engine.py --hf-repo {args.hf_repo} --output ./rfdetr_int8.engine
```

### Manual Download

```python
from huggingface_hub import hf_hub_download

onnx_path = hf_hub_download("{args.hf_repo}", "inference_model.onnx")
cache_path = hf_hub_download("{args.hf_repo}", "calibration.cache")
```

## Model Details

- **Architecture**: RF-DETR {args.model_variant.capitalize()}
- **Input Resolution**: 512x512
- **Input Format**: NCHW, normalized with ImageNet mean/std
- **Output Format**:
  - `dets`: [1, 300, 4] bounding boxes in cxcywh format (normalized 0-1)
  - `labels`: [1, 300, 91] class logits (apply sigmoid for scores)

## License

This model is distributed under the same license as RF-DETR.
"""
            print("Uploading README.md...")
            api.upload_file(
                path_or_fileobj=readme_content.encode(),
                path_in_repo="README.md",
                repo_id=args.hf_repo,
            )

            print(f"\nUpload complete! View at: https://huggingface.co/{args.hf_repo}")
    else:
        print("\nSkipping HuggingFace upload")

    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"\nArtifacts:")
    print(f"  ONNX model: {onnx_path}")
    print(f"  Calibration cache: {cache_path}")
    if not args.skip_upload and not args.dry_run:
        print(f"  HuggingFace: https://huggingface.co/{args.hf_repo}")


if __name__ == "__main__":
    main()
