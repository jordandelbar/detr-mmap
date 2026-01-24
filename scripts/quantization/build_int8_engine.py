#!/usr/bin/env python3
"""Build INT8 TensorRT engine using calibration tensors."""

import numpy as np
import tensorrt as trt
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).parent.parent.parent
CALIBRATION_DIR = Path(__file__).parent / "calibration_tensors"
MODEL_PATH = WORKSPACE_ROOT / "models" / "rfdetr_S" / "rfdetr.onnx"
OUTPUT_PATH = WORKSPACE_ROOT / "models" / "rfdetr_S" / "rfdetr_int8.engine"

# Input shape: NCHW format, matching preprocessing (see crates/preprocess/src/config.rs)
INPUT_SHAPE = (1, 3, 512, 512)


class CalibrationDataReader(trt.IInt8EntropyCalibrator2):
    def __init__(self, calibration_dir: Path, input_shape: tuple, cache_file: str = "calibration.cache"):
        super().__init__()
        self.cache_file = cache_file
        self.input_shape = input_shape
        self.batch_size = input_shape[0]

        # Load all calibration tensors
        self.calibration_files = sorted(calibration_dir.glob("*.bin"))
        self.num_samples = len(self.calibration_files)
        self.current_index = 0

        # Allocate device memory
        import pycuda.driver as cuda
        import pycuda.autoinit

        self.device_input = cuda.mem_alloc(
            int(np.prod(input_shape) * np.dtype(np.float32).itemsize)
        )

        print(f"Loaded {self.num_samples} calibration samples")

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_index >= self.num_samples:
            return None

        import pycuda.driver as cuda

        # Load calibration tensor
        tensor_path = self.calibration_files[self.current_index]
        tensor = np.fromfile(tensor_path, dtype=np.float32).reshape(self.input_shape)

        # Copy to device
        cuda.memcpy_htod(self.device_input, tensor.tobytes())

        self.current_index += 1
        return [int(self.device_input)]

    def read_calibration_cache(self):
        if Path(self.cache_file).exists():
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)


def build_engine():
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    # Parse ONNX model
    print(f"Parsing ONNX model: {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"Parser error: {parser.get_error(i)}")
            return None

    # Configure builder
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4GB

    # Enable INT8
    config.set_flag(trt.BuilderFlag.INT8)
    # Also keep FP16 as fallback for layers that don't support INT8
    config.set_flag(trt.BuilderFlag.FP16)

    # Set calibrator
    calibrator = CalibrationDataReader(
        CALIBRATION_DIR,
        INPUT_SHAPE,
        cache_file=str(Path(__file__).parent / "calibration.cache")
    )
    config.int8_calibrator = calibrator

    # Build engine
    print("Building INT8 engine (this may take a while)...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        print("Failed to build engine")
        return None

    # Save engine
    print(f"Saving engine to: {OUTPUT_PATH}")
    with open(OUTPUT_PATH, "wb") as f:
        f.write(serialized_engine)

    print("Done!")
    return serialized_engine


if __name__ == "__main__":
    build_engine()
