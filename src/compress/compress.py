"""Quantization runner script."""

import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from compress.quantize import Quantizer, QuantConfig, QuantMethod
from registry.manager import ModelRegistry, CheckpointInfo, EvalResult
from config import config


def quantize_model(
    model_path: str,
    method: QuantMethod,
    bits: int,
    registry_path: str,
) -> CheckpointInfo:
    """Quantize a model.

    Args:
        model_path: Path to model to quantize
        method: Quantization method (GGUF, AWQ, GPTQ)
        bits: Quantization bits (4 or 8)
        registry_path: Path to model registry

    Returns:
        Registered checkpoint info for quantized model
    """
    print("=" * 60)
    print(f"Model Quantization ({method.value.upper()}, {bits}-bit)")
    print("=" * 60)

    quant_config = QuantConfig(method=method, bits=bits)
    quantizer = Quantizer(quant_config)

    # Generate output path
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_name = f"{method.value}_{bits}bit"
    output_path = Path(f"artifacts/quantized/{output_name}-{timestamp}")
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Input: {model_path}")
    print(f"Output: {output_path}")
    print(f"Method: {method.value}")
    print(f"Bits: {bits}")

    # Quantize
    print("\nQuantizing...")
    manifest = quantizer.quantize(model_path, str(output_path))

    compression_ratio = manifest["compression_ratio"]
    size_before = manifest["size_before_mb"]
    size_after = manifest["size_after_mb"]

    print(f"\nQuantization complete!")
    print(f"Size before: {size_before:.0f} MB")
    print(f"Size after: {size_after:.0f} MB")
    print(f"Compression ratio: {compression_ratio:.2f}x")

    # Register checkpoint
    registry = ModelRegistry(registry_path)
    checkpoint_id = f"quantized-{output_name}-{timestamp}"

    checkpoint_info = CheckpointInfo(
        id=checkpoint_id,
        source=f"quantized-{method.value}",
        base_model=model_path,
        dataset_version=None,
        path=str(output_path),
        eval=None,  # Will be filled after evaluation
        is_best=False,
        created_at=datetime.now(),
        manifest=manifest,
    )

    registry.register_checkpoint(checkpoint_info)

    print(f"\nCheckpoint registered: {checkpoint_id}")
    print("Next step: Run evaluation to verify quality")

    return checkpoint_info


if __name__ == "__main__":
    # Example: Quantize to 4-bit GGUF
    checkpoint = quantize_model(
        model_path="artifacts/base/Qwen2.5-Coder-7B-Instruct",
        method=QuantMethod.GGUF,
        bits=4,
        registry_path=str(config.registry_path),
    )
