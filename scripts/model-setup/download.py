#!/usr/bin/env python3
"""Download ONNX model from HuggingFace.

For users who want to run the CPU version with docker-compose.

Usage:
    uv run download.py --hf-repo org/rf-detr-s-int8 --output ./models/
"""
import argparse
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download


def download_onnx(repo_id: str, output_dir: Path) -> Path:
    """Download ONNX model from HuggingFace and save to output directory."""
    print(f"Downloading ONNX from HuggingFace: {repo_id}")

    cached_path = hf_hub_download(repo_id, "inference_model.onnx")
    print(f"  Downloaded to cache: {cached_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "inference_model.onnx"

    shutil.copy(cached_path, output_path)
    print(f"  Copied to: {output_path}")

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nDone! ONNX model: {output_path} ({size_mb:.1f} MB)")

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download ONNX model from HuggingFace"
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        required=True,
        help="HuggingFace repository (e.g., 'org/rf-detr-s-int8')",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models"),
        help="Output directory (default: models)",
    )
    args = parser.parse_args()

    download_onnx(args.hf_repo, args.output)


if __name__ == "__main__":
    main()
