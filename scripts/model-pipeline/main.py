#!/usr/bin/env python3
"""Prepare and publish RF-DETR INT8 model to HuggingFace.

This is the main orchestration script for maintainers. It:
1. Exports the ONNX model (512x512)
2. Downloads calibration images
3. Generates calibration tensors (via Rust crate)
4. Builds calibration cache
5. Pushes ONNX + cache to HuggingFace
"""
import argparse
import asyncio
import subprocess
from pathlib import Path

from build_calibration_cache import build_calibration_cache
from download_calibration_data import download_calibration_images
from export_onnx import export_onnx

SCRIPT_DIR = Path(__file__).parent.resolve()
WORKSPACE_ROOT = SCRIPT_DIR.parent.parent


def print_step(name: str) -> None:
    """Print a step header."""
    print(f"\n{'='*60}")
    print(f"STEP: {name}")
    print(f"{'='*60}")


def run_cargo_calibration(
    input_dir: Path, output_dir: Path, count: int, dry_run: bool = False
) -> None:
    """Run the Rust calibration crate to generate preprocessed tensors."""
    print_step("Generate calibration tensors")

    cmd = [
        "cargo", "run", "-p", "calibration", "--release", "--",
        "--input-dir", str(input_dir),
        "--output-dir", str(output_dir),
        "--count", str(count),
    ]

    if dry_run:
        print(f"[DRY RUN] Would run: {' '.join(cmd)}")
        return

    subprocess.run(cmd, cwd=WORKSPACE_ROOT, check=True)


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
        print_step("Export ONNX model")
        if args.dry_run:
            print(f"[DRY RUN] Would export ONNX to {model_dir}")
        else:
            export_onnx(model_dir, args.model_variant)
    else:
        print(f"\nSkipping ONNX export (using existing: {onnx_path})")

    # Step 2: Download calibration images
    if not args.skip_calibration_download:
        print_step("Download calibration images")
        if args.dry_run:
            print(f"[DRY RUN] Would download {args.calibration_images} images to {calibration_data_dir}")
        else:
            asyncio.run(download_calibration_images(calibration_data_dir, args.calibration_images))
    else:
        print(f"\nSkipping calibration download (using existing: {calibration_data_dir})")

    # Step 3: Generate calibration tensors (Rust)
    run_cargo_calibration(
        calibration_data_dir,
        calibration_tensor_dir,
        args.calibration_images,
        args.dry_run,
    )

    # Step 4: Build calibration cache
    print_step("Build calibration cache")
    if args.dry_run:
        print(f"[DRY RUN] Would build cache at {cache_path}")
    else:
        build_calibration_cache(onnx_path, cache_path, calibration_tensor_dir)

    # Step 5: Upload to HuggingFace
    if not args.skip_upload:
        print_step("Upload to HuggingFace")

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
            readme_content = f"""---
license: apache-2.0
tags:
  - object-detection
  - tensorrt
  - int8
  - onnx
pipeline_tag: object-detection
---

# RF-DETR {args.model_variant.capitalize()} INT8

Pre-exported ONNX model and TensorRT INT8 calibration cache for **RF-DETR {args.model_variant.capitalize()}**.

## Files
- `inference_model.onnx` — FP32 ONNX model (512×512)
- `calibration.cache` — TensorRT INT8 calibration cache

## Usage

```python
from huggingface_hub import hf_hub_download

onnx = hf_hub_download("org/rfdetr-small-int8", "inference_model.onnx")
cache = hf_hub_download("org/rfdetr-small-int8", "calibration.cache")
```
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
