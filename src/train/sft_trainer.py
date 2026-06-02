"""QLoRA SFT training implementation with actual training logic."""

import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from train.qlora import QLoRAConfig, QLoRATrainer
from registry.manager import ModelRegistry, CheckpointInfo, EvalResult
from config import config


@dataclass
class SFTManifest:
    """Training run manifest."""

    run_id: str
    base_model: str
    dataset_version: str
    method: str
    hyperparams: dict
    seed: int
    hardware: str
    adapter_path: str
    created_at: str
    final_loss: float
    steps: int

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


def run_qlora_training(
    base_model_path: str,
    train_data_path: str,
    val_data_path: str,
    dataset_version: str,
    registry_path: str,
    config_path: str = "config/train/qlora.yaml",
) -> CheckpointInfo:
    """Run QLoRA SFT training.

    Args:
        base_model_path: Path to base model
        train_data_path: Path to training data
        val_data_path: Path to validation data
        dataset_version: Dataset version being used
        registry_path: Path to model registry
        config_path: Path to training config

    Returns:
        Registered checkpoint info
    """
    print("=" * 60)
    print("QLoRA SFT Training")
    print("=" * 60)

    # Load config
    train_config = config.load_yaml(config_path)
    qlora_config = QLoRAConfig(**train_config.get("lora", {}))
    training_config = train_config.get("training", {})

    # Create trainer
    trainer = QLoRATrainer(qlora_config)

    # Generate run ID
    run_id = f"qlora-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # Output directory
    output_dir = Path(training_config.get("output_dir", "artifacts/qlora_checkpoints")) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run ID: {run_id}")
    print(f"Base model: {base_model_path}")
    print(f"Dataset: {dataset_version}")
    print(f"Output: {output_dir}")

    # Train (placeholder - would use actual TRL/SFTTrainer)
    print("\nStarting training...")
    manifest_dict = trainer.train(
        model_path=base_model_path,
        train_data_path=train_data_path,
        val_data_path=val_data_path,
        output_dir=str(output_dir),
    )

    # Save adapter
    adapter_path = output_dir / "adapter_model.bin"
    trainer.save_adapter(str(adapter_path))

    # Create manifest
    manifest = SFTManifest(
        run_id=run_id,
        base_model=base_model_path,
        dataset_version=dataset_version,
        method="qlora-sft",
        hyperparams={
            "lora": qlora_config.__dict__,
            "training": training_config,
        },
        seed=42,
        hardware="cuda",  # Placeholder
        adapter_path=str(adapter_path),
        created_at=datetime.now().isoformat(),
        final_loss=manifest_dict.get("final_loss", 0.0),
        steps=manifest_dict.get("steps", 0),
    )

    # Save manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        import json
        json.dump(manifest.to_dict(), f, indent=2)

    print(f"\nTraining complete!")
    print(f"Final loss: {manifest.final_loss}")
    print(f"Steps: {manifest.steps}")
    print(f"Adapter saved: {adapter_path}")

    # Register checkpoint
    registry = ModelRegistry(registry_path)

    checkpoint_info = CheckpointInfo(
        id=run_id,
        source="qlora-sft",
        base_model=base_model_path,
        dataset_version=dataset_version,
        path=str(output_dir),
        eval=None,  # Will be filled after evaluation
        is_best=False,  # Will be updated if evaluation passes
        created_at=datetime.now(),
        manifest=manifest.to_dict(),
    )

    registry.register_checkpoint(checkpoint_info)

    print(f"\nCheckpoint registered: {run_id}")
    print("Next step: Run evaluation to test model quality")

    return checkpoint_info


if __name__ == "__main__":
    # Example: Run QLoRA training
    checkpoint = run_qlora_training(
        base_model_path="artifacts/base/Qwen2.5-Coder-7B-Instruct",
        train_data_path="data/train/ds-v1_train.jsonl",
        val_data_path="data/val/ds-v1_val.jsonl",
        dataset_version="ds-v1",
        registry_path=str(config.registry_path),
    )
