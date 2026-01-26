#!/usr/bin/env python3
"""Export RF-DETR model to ONNX format.

This script clones the rf-detr repository if needed and exports the model
to ONNX format at 512x512 resolution for INT8 quantization.
"""
import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
WORKSPACE_ROOT = SCRIPT_DIR.parent.parent
RF_DETR_REPO = "https://github.com/roboflow/rf-detr.git"
RF_DETR_VERSION = "v1.4.0"  # Pin to specific release
RF_DETR_DIR = WORKSPACE_ROOT / "rf-detr"

# Fixed resolution for INT8 pipeline
RESOLUTION = 512


def clone_rf_detr() -> None:
    """Clone rf-detr repository and install it if needed."""
    if not RF_DETR_DIR.exists():
        print(f"Cloning rf-detr {RF_DETR_VERSION} to {RF_DETR_DIR}...")
        subprocess.run(
            [
                "git", "clone",
                "--branch", RF_DETR_VERSION,
                "--depth", "1",
                RF_DETR_REPO,
                str(RF_DETR_DIR),
            ],
            check=True,
        )
        print("Clone completed.")
    else:
        print(f"rf-detr already exists at {RF_DETR_DIR}")

    # Always ensure rf-detr is installed (uv pip install -e is idempotent)
    # Use --python to target the current interpreter's environment
    # Pin transformers<5 due to breaking API changes in v5.0
    print("Installing rf-detr dependencies...")
    subprocess.run(
        [
            "uv", "pip", "install", "--python", sys.executable,
            "-e", f"{RF_DETR_DIR}[onnxexport]",
            "transformers<5",
        ],
        check=True,
    )
    print("rf-detr installed.")


def export_onnx(output_dir: Path, model_variant: str) -> Path:
    """Export RF-DETR model to ONNX format.

    Args:
        output_dir: Directory to save the ONNX model.
        model_variant: Model variant to export ('small' or 'base').

    Returns:
        Path to the exported ONNX file.
    """
    import torch
    from rfdetr import RFDETRSmall, RFDETRBase

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model based on variant
    print(f"Loading RF-DETR {model_variant.capitalize()} model...")
    if model_variant == "small":
        model = RFDETRSmall(resolution=RESOLUTION)
    elif model_variant == "base":
        model = RFDETRBase(resolution=RESOLUTION)
    else:
        raise ValueError(f"Unknown model variant: {model_variant}")

    print(f"Model loaded with resolution: {model.model.resolution}")
    print(f"Number of classes: {model.model_config.num_classes}")

    # Get the internal model and prepare for export
    internal_model = model.model.model
    internal_model.eval()

    # Call export() on the model to prepare it for ONNX export
    if hasattr(internal_model, "export"):
        internal_model.export()

    # Create dummy input
    dummy_input = torch.randn(1, 3, RESOLUTION, RESOLUTION)

    print(f"\nExporting to ONNX in {output_dir}...")
    print(f"Input shape: {dummy_input.shape}")

    # Run inference to check output shapes
    with torch.no_grad():
        outputs = internal_model(dummy_input.cuda())
        if isinstance(outputs, dict):
            print(
                f"Output shapes - boxes: {outputs['pred_boxes'].shape}, "
                f"logits: {outputs['pred_logits'].shape}"
            )

    # Export to ONNX
    output_file = output_dir / "inference_model.onnx"

    internal_model.cpu()
    dummy_input = dummy_input.cpu()

    torch.onnx.export(
        internal_model,
        dummy_input,
        str(output_file),
        input_names=["input"],
        output_names=["dets", "labels"],
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        verbose=False,
        dynamo=False,  # Use legacy ONNX exporter
    )

    print(f"\nONNX export completed!")
    print(f"Output file: {output_file}")
    print(f"\nOutput format:")
    print(f"  - 'dets': pred_boxes [1, 300, 4] in cxcywh format (normalized 0-1)")
    print(f"  - 'labels': pred_logits [1, 300, 91] (raw logits, apply sigmoid for scores)")

    return output_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export RF-DETR model to ONNX format"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "models" / "rfdetr_small",
        help="Output directory for the ONNX model",
    )
    parser.add_argument(
        "--model-variant",
        choices=["small", "base"],
        default="small",
        help="Model variant to export (default: small)",
    )
    parser.add_argument(
        "--skip-clone",
        action="store_true",
        help="Skip cloning rf-detr repository",
    )
    args = parser.parse_args()

    if not args.skip_clone:
        clone_rf_detr()
        # Re-exec with --skip-clone so the newly installed rfdetr is importable
        subprocess.run(
            [
                sys.executable,
                __file__,
                "--output-dir", str(args.output_dir),
                "--model-variant", args.model_variant,
                "--skip-clone",
            ],
            check=True,
        )
        return

    export_onnx(args.output_dir, args.model_variant)


if __name__ == "__main__":
    main()
