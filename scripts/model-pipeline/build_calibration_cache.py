#!/usr/bin/env python3
"""Build INT8 calibration cache from preprocessed tensors.

This script builds a TensorRT INT8 calibration cache that can be distributed
to users for building their own TensorRT engines without needing to repeat
the calibration process.
"""
import argparse
import subprocess
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).parent.resolve()
WORKSPACE_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_CALIBRATION_DIR = SCRIPT_DIR / "calibration_tensors"
DEFAULT_IMAGE_DIR = SCRIPT_DIR / "calibration_data"

# Input shape: NCHW format, matching preprocessing
INPUT_SHAPE = (1, 3, 512, 512)


class CacheOnlyCalibrator:
    """Calibrator that only builds the cache, doesn't build an engine."""

    def __init__(self, calibration_dir: Path, cache_file: Path, input_shape: tuple):
        import tensorrt as trt

        # Store as instance variable to avoid issues
        self.trt = trt
        self.cache_file = cache_file
        self.input_shape = input_shape
        self.batch_size = input_shape[0]

        # Load all calibration tensors
        self.calibration_files = sorted(calibration_dir.glob("*.bin"))
        self.num_samples = len(self.calibration_files)
        self.current_index = 0

        if self.num_samples == 0:
            raise RuntimeError(
                f"No calibration tensors found in {calibration_dir}. "
                "Run the calibration crate first to generate tensors."
            )

        # Allocate device memory
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401

        self.device_input = cuda.mem_alloc(
            int(np.prod(input_shape) * np.dtype(np.float32).itemsize)
        )

        print(f"Loaded {self.num_samples} calibration samples from {calibration_dir}")

    def get_batch_size(self) -> int:
        return self.batch_size

    def get_batch(self, names: list) -> list | None:
        if self.current_index >= self.num_samples:
            return None

        import pycuda.driver as cuda

        # Load calibration tensor
        tensor_path = self.calibration_files[self.current_index]
        tensor = np.fromfile(tensor_path, dtype=np.float32).reshape(self.input_shape)

        # Copy to device
        cuda.memcpy_htod(self.device_input, tensor.tobytes())

        self.current_index += 1
        if self.current_index % 10 == 0:
            print(f"  Calibrating... {self.current_index}/{self.num_samples}")

        return [int(self.device_input)]

    def read_calibration_cache(self) -> bytes | None:
        if self.cache_file.exists():
            print(f"Loading existing cache from {self.cache_file}")
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache: bytes) -> None:
        print(f"Writing calibration cache to {self.cache_file}")
        with open(self.cache_file, "wb") as f:
            f.write(cache)


def run_cargo_calibration(
    input_dir: Path, output_dir: Path, count: int
) -> None:
    """Run the Rust calibration crate to generate preprocessed tensors."""
    print(f"Running cargo calibration with {count} images...")

    cmd = [
        "cargo", "run", "-p", "calibration", "--release", "--",
        "--input-dir", str(input_dir),
        "--output-dir", str(output_dir),
        "--count", str(count),
    ]

    subprocess.run(cmd, cwd=WORKSPACE_ROOT, check=True)
    print("Calibration tensors generated successfully.")


def build_calibration_cache(
    onnx_path: Path,
    cache_path: Path,
    calibration_dir: Path,
) -> None:
    """Build INT8 calibration cache.

    Args:
        onnx_path: Path to the ONNX model.
        cache_path: Path to save the calibration cache.
        calibration_dir: Directory containing calibration tensors.
    """
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    # Parse ONNX model
    print(f"Parsing ONNX model: {onnx_path}")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"Parser error: {parser.get_error(i)}")
            raise RuntimeError("Failed to parse ONNX model")

    # Configure builder for calibration
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4GB

    # Enable INT8 and FP16
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)

    # Create calibrator
    calibrator = CacheOnlyCalibrator(
        calibration_dir,
        cache_path,
        INPUT_SHAPE,
    )

    # Make the calibrator a proper IInt8EntropyCalibrator2
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

    config.int8_calibrator = TRTCalibrator(calibrator)

    # Build engine (this generates the calibration cache as a side effect)
    print("Building calibration cache (this may take a while)...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        raise RuntimeError("Failed to build calibration cache")

    print(f"\nCalibration cache saved to: {cache_path}")
    print(f"Cache size: {cache_path.stat().st_size / 1024:.1f} KB")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build INT8 calibration cache for RF-DETR"
    )
    parser.add_argument(
        "--onnx-path",
        type=Path,
        required=True,
        help="Path to the ONNX model",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=Path("calibration.cache"),
        help="Output path for the calibration cache (default: calibration.cache)",
    )
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=DEFAULT_CALIBRATION_DIR,
        help=f"Directory with calibration tensors (default: {DEFAULT_CALIBRATION_DIR})",
    )
    parser.add_argument(
        "--run-cargo-calibration",
        action="store_true",
        help="Run cargo calibration first to generate tensors",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=DEFAULT_IMAGE_DIR,
        help=f"Directory with calibration images (default: {DEFAULT_IMAGE_DIR})",
    )
    parser.add_argument(
        "--image-count",
        type=int,
        default=100,
        help="Number of images to use for calibration (default: 100)",
    )
    args = parser.parse_args()

    if args.run_cargo_calibration:
        run_cargo_calibration(args.image_dir, args.calibration_dir, args.image_count)

    build_calibration_cache(args.onnx_path, args.cache_path, args.calibration_dir)


if __name__ == "__main__":
    main()
