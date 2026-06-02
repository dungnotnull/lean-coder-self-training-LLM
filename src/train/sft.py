"""Supervised Fine-Tuning (SFT) implementation."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class SFTConfig:
    """SFT training configuration."""

    learning_rate: float = 2e-5
    num_epochs: int = 3
    batch_size: int = 4
    max_length: int = 2048
    warmup_steps: int = 100


class SFTTrainer:
    """Supervised fine-tuning trainer."""

    def __init__(self, config: SFTConfig):
        """Initialize SFT trainer.

        Args:
            config: SFT configuration
        """
        self.config = config

    def train(
        self,
        model_path: str,
        train_data_path: str,
        val_data_path: str,
        output_dir: str,
    ) -> dict:
        """Run SFT training.

        Args:
            model_path: Path to base model
            train_data_path: Path to training data
            val_data_path: Path to validation data
            output_dir: Path for output checkpoints

        Returns:
            Training manifest with metrics
        """
        manifest = {
            "method": "sft",
            "base_model": model_path,
            "train_data": train_data_path,
            "val_data": val_data_path,
            "output_dir": output_dir,
            "config": self.config.__dict__,
            "final_loss": 0.0,
            "steps": 0,
        }

        return manifest
