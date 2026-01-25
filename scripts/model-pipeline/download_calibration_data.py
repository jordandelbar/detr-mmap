#!/usr/bin/env python3
"""Download COCO validation images for INT8 calibration.

This script downloads random images from the COCO 2017 validation set
to be used for INT8 calibration of the RF-DETR model.
"""
import argparse
import random
from pathlib import Path

import requests

SCRIPT_DIR = Path(__file__).parent.resolve()
WORKSPACE_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_OUTPUT_DIR = WORKSPACE_ROOT / "scripts" / "quantization" / "calibration_data"

COCO_VAL_URL = "http://images.cocodataset.org/val2017/"
MAX_COCO_ID = 600_000  # Safe upper bound for COCO image IDs
TIMEOUT = 10


def download_calibration_images(output_dir: Path, count: int, seed: int = 42) -> int:
    """Download COCO validation images for calibration.

    Args:
        output_dir: Directory to save downloaded images.
        count: Number of images to download.
        seed: Random seed for reproducibility.

    Returns:
        Number of images successfully downloaded.
    """
    random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    attempts = 0
    max_attempts = count * 10  # Avoid infinite loop

    print(f"Downloading {count} COCO validation images to {output_dir}...")

    while downloaded < count and attempts < max_attempts:
        img_id = random.randint(1, MAX_COCO_ID)
        img_name = f"{img_id:012d}.jpg"
        img_path = output_dir / img_name

        if img_path.exists():
            continue

        img_url = f"{COCO_VAL_URL}{img_name}"
        attempts += 1

        try:
            response = requests.get(img_url, timeout=TIMEOUT)
            if response.status_code == 200:
                img_path.write_bytes(response.content)
                downloaded += 1
                print(f"[{downloaded}/{count}] Downloaded {img_name}")
        except requests.RequestException:
            pass  # Ignore 404 / network errors

    print(f"\nDone. Downloaded {downloaded} images in {attempts} attempts.")
    return downloaded


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download COCO images for INT8 calibration"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for images (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Number of images to download (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    download_calibration_images(args.output_dir, args.count, args.seed)


if __name__ == "__main__":
    main()
