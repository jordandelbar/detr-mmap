#!/usr/bin/env python3
"""Build a TensorRT INT8 engine from HuggingFace-hosted model and cache.

This is a standalone script that users can run to build a TensorRT INT8 engine
for their specific GPU. It downloads the ONNX model and calibration cache from
HuggingFace and builds the engine locally.

Usage:
    python user_build_engine.py --hf-repo your-org/rfdetr-small-int8 --output ./rfdetr_int8.engine

Requirements:
    pip install huggingface_hub tensorrt pycuda numpy
"""
import argparse
import sys
from pathlib import Path

# Input shape: NCHW format, must match the calibration cache
INPUT_SHAPE = (1, 3, 512, 512)


def download_from_hf(repo_id: str, cache_dir: Path | None = None) -> tuple[Path, Path]:
    """Download ONNX model and calibration cache from HuggingFace.

    Args:
        repo_id: HuggingFace repository ID (e.g., 'your-org/rfdetr-small-int8').
        cache_dir: Optional directory to cache downloaded files.

    Returns:
        Tuple of (onnx_path, cache_path).
    """
    from huggingface_hub import hf_hub_download

    print(f"Downloading from HuggingFace: {repo_id}")

    onnx_path = hf_hub_download(
        repo_id,
        "inference_model.onnx",
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    print(f"  Downloaded: inference_model.onnx")

    cache_path = hf_hub_download(
        repo_id,
        "calibration.cache",
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    print(f"  Downloaded: calibration.cache")

    return Path(onnx_path), Path(cache_path)


class CacheReadingCalibrator:
    """Calibrator that reads from an existing cache file (no data collection)."""

    def __init__(self, cache_file: Path):
        self.cache_file = cache_file

        if not cache_file.exists():
            raise FileNotFoundError(f"Calibration cache not found: {cache_file}")

        print(f"Using calibration cache: {cache_file}")

    def get_batch_size(self) -> int:
        return INPUT_SHAPE[0]

    def get_batch(self, names: list) -> None:
        # No data collection needed when using cached calibration
        return None

    def read_calibration_cache(self) -> bytes:
        with open(self.cache_file, "rb") as f:
            return f.read()

    def write_calibration_cache(self, cache: bytes) -> None:
        # Cache already exists, no need to write
        pass


def build_engine(onnx_path: Path, cache_path: Path, output_path: Path) -> None:
    """Build TensorRT INT8 engine using cached calibration data.

    Args:
        onnx_path: Path to the ONNX model.
        cache_path: Path to the calibration cache.
        output_path: Path to save the built engine.
    """
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    # Parse ONNX model
    print(f"\nParsing ONNX model: {onnx_path}")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"Parser error: {parser.get_error(i)}")
            sys.exit(1)

    # Configure builder
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4GB

    # Enable INT8 and FP16
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)

    # Create calibrator that reads from cache
    inner_calibrator = CacheReadingCalibrator(cache_path)

    # Wrap in TensorRT calibrator interface
    class TRTCalibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def get_batch_size(self):
            return self.inner.get_batch_size()

        def get_batch(self, names):
            return self.inner.get_batch(names)

        def read_calibration_cache(self):
            return self.inner.read_calibration_cache()

        def write_calibration_cache(self, cache):
            return self.inner.write_calibration_cache(cache)

    config.int8_calibrator = TRTCalibrator(inner_calibrator)

    # Build engine
    print("\nBuilding TensorRT INT8 engine (this may take several minutes)...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        print("Failed to build engine")
        sys.exit(1)

    # Save engine
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving engine to: {output_path}")
    with open(output_path, "wb") as f:
        f.write(serialized_engine)

    engine_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Engine size: {engine_size_mb:.1f} MB")
    print("\nDone! Your INT8 engine is ready to use.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build TensorRT INT8 engine from HuggingFace-hosted model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Build engine from default HuggingFace repo
    python user_build_engine.py --hf-repo your-org/rfdetr-small-int8 --output ./rfdetr_int8.engine

    # Use local files instead of downloading
    python user_build_engine.py --onnx ./model.onnx --cache ./calibration.cache --output ./engine.engine
""",
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        help="HuggingFace repository ID (e.g., 'your-org/rfdetr-small-int8')",
    )
    parser.add_argument(
        "--onnx",
        type=Path,
        help="Path to local ONNX model (alternative to --hf-repo)",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        help="Path to local calibration cache (alternative to --hf-repo)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("rfdetr_int8.engine"),
        help="Output path for the TensorRT engine (default: rfdetr_int8.engine)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Directory to cache downloaded files from HuggingFace",
    )
    args = parser.parse_args()

    # Validate arguments
    if args.hf_repo and (args.onnx or args.cache):
        parser.error("Cannot specify both --hf-repo and --onnx/--cache")

    if not args.hf_repo and not (args.onnx and args.cache):
        parser.error("Must specify either --hf-repo or both --onnx and --cache")

    # Get model files
    if args.hf_repo:
        onnx_path, cache_path = download_from_hf(args.hf_repo, args.cache_dir)
    else:
        onnx_path = args.onnx
        cache_path = args.cache

        if not onnx_path.exists():
            print(f"Error: ONNX file not found: {onnx_path}")
            sys.exit(1)
        if not cache_path.exists():
            print(f"Error: Cache file not found: {cache_path}")
            sys.exit(1)

    # Build engine
    build_engine(onnx_path, cache_path, args.output)


if __name__ == "__main__":
    main()
