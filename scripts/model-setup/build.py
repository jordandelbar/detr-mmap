#!/usr/bin/env python3
"""Build TensorRT INT8 engine from HuggingFace-hosted model and cache.

For users with NVIDIA GPU who want to run the k3d/TensorRT setup.
Requires: NVIDIA drivers, CUDA, TensorRT.

Usage:
    uv run build.py --hf-repo org/rf-detr-s-int8 --output ./models/rfdetr_int8.engine
"""
import argparse
import sys
from pathlib import Path

import tensorrt as trt
from huggingface_hub import hf_hub_download

# Input shape must match the calibration cache
INPUT_SHAPE = (1, 3, 512, 512)


def download_from_hf(repo_id: str) -> tuple[Path, Path]:
    """Download ONNX model and calibration cache from HuggingFace."""
    print(f"Downloading from HuggingFace: {repo_id}")

    onnx_path = hf_hub_download(repo_id, "inference_model.onnx")
    print("  Downloaded: inference_model.onnx")

    cache_path = hf_hub_download(repo_id, "calibration.cache")
    print("  Downloaded: calibration.cache")

    return Path(onnx_path), Path(cache_path)


class CachedCalibrator(trt.IInt8EntropyCalibrator2):
    """Calibrator that reads from an existing cache file."""

    def __init__(self, cache_file: Path):
        super().__init__()
        self.cache_file = cache_file

        if not cache_file.exists():
            raise FileNotFoundError(f"Calibration cache not found: {cache_file}")

        print(f"Using calibration cache: {cache_file}")

    def get_batch_size(self) -> int:
        return INPUT_SHAPE[0]

    def get_batch(self, names: list) -> None:
        return None

    def read_calibration_cache(self) -> bytes:
        return self.cache_file.read_bytes()

    def write_calibration_cache(self, cache: bytes) -> None:
        pass


def build_engine(onnx_path: Path, cache_path: Path, output_path: Path) -> None:
    """Build TensorRT INT8 engine using cached calibration data."""
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    print(f"\nParsing ONNX model: {onnx_path}")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"Parser error: {parser.get_error(i)}")
            sys.exit(1)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4GB
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)
    config.int8_calibrator = CachedCalibrator(cache_path)

    print("\nBuilding TensorRT INT8 engine...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        print("Failed to build engine")
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving engine to: {output_path}")
    output_path.write_bytes(serialized_engine)

    engine_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Engine size: {engine_size_mb:.1f} MB")
    print("\nDone!")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build TensorRT INT8 engine from HuggingFace model"
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        help="HuggingFace repository (e.g., 'org/rf-detr-s-int8')",
    )
    parser.add_argument(
        "--onnx",
        type=Path,
        help="Local ONNX model path (alternative to --hf-repo)",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        help="Local calibration cache path (alternative to --hf-repo)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("rfdetr_int8.engine"),
        help="Output engine path (default: rfdetr_int8.engine)",
    )
    args = parser.parse_args()

    if args.hf_repo and (args.onnx or args.cache):
        parser.error("Cannot specify both --hf-repo and --onnx/--cache")

    if not args.hf_repo and not (args.onnx and args.cache):
        parser.error("Must specify either --hf-repo or both --onnx and --cache")

    if args.hf_repo:
        onnx_path, cache_path = download_from_hf(args.hf_repo)
    else:
        onnx_path, cache_path = args.onnx, args.cache
        if not onnx_path.exists():
            print(f"Error: ONNX file not found: {onnx_path}")
            sys.exit(1)
        if not cache_path.exists():
            print(f"Error: Cache file not found: {cache_path}")
            sys.exit(1)

    build_engine(onnx_path, cache_path, args.output)


if __name__ == "__main__":
    main()
