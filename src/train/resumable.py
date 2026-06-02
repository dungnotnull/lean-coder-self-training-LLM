"""Resumable training with checkpoint saving and restoration."""

import os
import json
import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class TrainingState:
    """Complete training state for resumption."""

    checkpoint_id: str
    current_step: int
    current_epoch: int
    best_loss: float
    best_eval_score: float
    global_step: int
    optimizer_state: Dict
    scheduler_state: Optional[Dict]
    lr_scheduler_state: Optional[Dict]
    random_states: Dict[str, Any]
    training_args: Dict
    created_at: str
    updated_at: str

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "current_step": self.current_step,
            "current_epoch": self.current_epoch,
            "best_loss": self.best_loss,
            "best_eval_score": self.best_eval_score,
            "global_step": self.global_step,
            "optimizer_state": self.optimizer_state,
            "scheduler_state": self.scheduler_state,
            "lr_scheduler_state": self.lr_scheduler_state,
            "random_states": self.random_states,
            "training_args": self.training_args,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class ResumableTrainer:
    """Trainer with resumable training capabilities."""

    def __init__(
        self,
        output_dir: str,
        checkpoint_interval: int = 100,
        save_total_limit: int = 3,
    ):
        """Initialize resumable trainer.

        Args:
            output_dir: Directory for checkpoints
            checkpoint_interval: Save checkpoint every N steps
            save_total_limit: Maximum number of checkpoints to keep
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_interval = checkpoint_interval
        self.save_total_limit = save_total_limit

        self.training_state: Optional[TrainingState] = None
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True)

    def initialize_training(
        self,
        checkpoint_id: str,
        training_args: Dict,
    ) -> TrainingState:
        """Initialize a new training run.

        Args:
            checkpoint_id: Unique checkpoint ID
            training_args: Training hyperparameters

        Returns:
            Initial TrainingState
        """
        now = datetime.now().isoformat()

        self.training_state = TrainingState(
            checkpoint_id=checkpoint_id,
            current_step=0,
            current_epoch=0,
            best_loss=float("inf"),
            best_eval_score=0.0,
            global_step=0,
            optimizer_state={},
            scheduler_state=None,
            lr_scheduler_state=None,
            random_states={},
            training_args=training_args,
            created_at=now,
            updated_at=now,
        )

        self._save_state()
        return self.training_state

    def save_checkpoint(
        self,
        model,
        optimizer,
        lr_scheduler=None,
        scheduler=None,
        metrics: Dict = None,
    ):
        """Save training checkpoint.

        Args:
            model: Model to save
            optimizer: Optimizer state
            lr_scheduler: Learning rate scheduler
            scheduler: Training scheduler
            metrics: Optional metrics dict
        """
        if self.training_state is None:
            raise ValueError("Training not initialized. Call initialize_training() first.")

        checkpoint_dir = self.checkpoints_dir / f"step-{self.training_state.global_step}"
        checkpoint_dir.mkdir(exist_ok=True)

        # Save model
        torch.save(model.state_dict(), checkpoint_dir / "model.pt")

        # Save optimizer
        self.training_state.optimizer_state = {
            "state_dict": optimizer.state_dict(),
            "param_groups": optimizer.param_groups,
        }

        # Save schedulers
        if lr_scheduler:
            self.training_state.lr_scheduler_state = {
                "state_dict": lr_scheduler.state_dict(),
            }

        if scheduler:
            self.training_state.scheduler_state = {
                "state_dict": scheduler.state_dict(),
            }

        # Save random states
        self.training_state.random_states = {
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state() if 'np' in dir() else None,
        }

        # Update metrics
        if metrics:
            self.training_state.best_loss = min(
                self.training_state.best_loss,
                metrics.get("loss", self.training_state.best_loss),
            )
            self.training_state.best_eval_score = max(
                self.training_state.best_eval_score,
                metrics.get("eval_score", self.training_state.best_eval_score),
            )

        # Update timestamp
        self.training_state.updated_at = datetime.now().isoformat()

        # Save state
        self._save_state()

        # Clean old checkpoints
        self._cleanup_old_checkpoints()

        logger.info(f"Checkpoint saved at step {self.training_state.global_step}")

    def restore_checkpoint(
        self,
        checkpoint_id: str = None,
        step: int = None,
    ) -> TrainingState:
        """Restore from checkpoint.

        Args:
            checkpoint_id: Specific checkpoint ID
            step: Specific step number

        Returns:
            Restored TrainingState
        """
        if checkpoint_id:
            state_file = self.checkpoints_dir / checkpoint_id / "training_state.json"
        elif step:
            state_file = self.checkpoints_dir / f"step-{step}" / "training_state.json"
        else:
            # Find latest checkpoint
            checkpoints = list(self.checkpoints_dir.glob("step-*"))
            if not checkpoints:
                raise ValueError("No checkpoints found")
            state_file = max(checkpoints, key=lambda p: p.stat().st_mtime) / "training_state.json"

        if not state_file.exists():
            raise ValueError(f"Checkpoint not found: {state_file}")

        with open(state_file) as f:
            state_data = json.load(f)

        self.training_state = TrainingState(**state_data)

        logger.info(f"Restored checkpoint from step {self.training_state.global_step}")

        return self.training_state

    def load_model_state(self, model, checkpoint_id: str = None, step: int = None):
        """Load model state from checkpoint.

        Args:
            model: Model to load state into
            checkpoint_id: Specific checkpoint ID
            step: Specific step number
        """
        if checkpoint_id:
            model_file = self.checkpoints_dir / checkpoint_id / "model.pt"
        elif step:
            model_file = self.checkpoints_dir / f"step-{step}" / "model.pt"
        else:
            if self.training_state is None:
                raise ValueError("No training state. Call restore_checkpoint() first.")
            model_file = self.checkpoints_dir / f"step-{self.training_state.global_step}" / "model.pt"

        if not model_file.exists():
            raise ValueError(f"Model checkpoint not found: {model_file}")

        model.load_state_dict(torch.load(model_file))
        logger.info(f"Model state loaded from {model_file}")

    def load_optimizer_state(self, optimizer, checkpoint_id: str = None, step: int = None):
        """Load optimizer state from checkpoint.

        Args:
            optimizer: Optimizer to load state into
            checkpoint_id: Specific checkpoint ID
            step: Specific step number
        """
        if self.training_state is None:
            raise ValueError("No training state. Call restore_checkpoint() first.")

        if checkpoint_id:
            state_file = self.checkpoints_dir / checkpoint_id / "training_state.json"
        elif step:
            state_file = self.checkpoints_dir / f"step-{step}" / "training_state.json"
        else:
            state_file = self.checkpoints_dir / f"step-{self.training_state.global_step}" / "training_state.json"

        if not state_file.exists():
            raise ValueError(f"Checkpoint not found: {state_file}")

        with open(state_file) as f:
            state_data = json.load(f)

        opt_state = state_data.get("optimizer_state", {})
        if opt_state:
            optimizer.load_state_dict(opt_state["state_dict"])
            logger.info("Optimizer state loaded")

    def step(self):
        """Increment training step."""
        if self.training_state is None:
            raise ValueError("Training not initialized.")
        self.training_state.current_step += 1
        self.training_state.global_step += 1

    def epoch_step(self):
        """Increment training epoch."""
        if self.training_state is None:
            raise ValueError("Training not initialized.")
        self.training_state.current_epoch += 1

    def should_save_checkpoint(self) -> bool:
        """Check if should save checkpoint.

        Returns:
            True if checkpoint interval reached
        """
        if self.training_state is None:
            return False
        return self.training_state.global_step % self.checkpoint_interval == 0

    def _save_state(self):
        """Save training state to JSON."""
        state_file = self.checkpoints_dir / "training_state.json"
        with open(state_file, "w") as f:
            json.dump(self.training_state.to_dict(), f, indent=2)

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints beyond save limit."""
        checkpoints = sorted(
            self.checkpoints_dir.glob("step-*"),
            key=lambda p: int(p.name.split("-")[1]),
        )

        while len(checkpoints) > self.save_total_limit:
            oldest = checkpoints.pop(0)
            # Remove oldest checkpoint directory
            for file in oldest.glob("*"):
                file.unlink()
            oldest.rmdir()
            logger.info(f"Removed old checkpoint: {oldest.name}")


class ProgressTracker:
    """Track and display training progress."""

    def __init__(self, total_steps: int, log_interval: int = 10):
        """Initialize progress tracker.

        Args:
            total_steps: Total number of training steps
            log_interval: Log progress every N steps
        """
        self.total_steps = total_steps
        self.log_interval = log_interval
        self.current_step = 0
        self.start_time = None

    def start(self):
        """Start tracking."""
        self.start_time = datetime.now()
        print(f"\nTraining started: {self.total_steps} steps")
        print("=" * 60)

    def update(self, step: int, metrics: Dict = None):
        """Update progress.

        Args:
            step: Current step number
            metrics: Optional metrics to display
        """
        self.current_step = step

        if step % self.log_interval == 0 or step == self.total_steps:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            speed = step / elapsed if elapsed > 0 else 0
            remaining = (self.total_steps - step) / speed if speed > 0 else 0

            progress = (step / self.total_steps) * 100
            bar_length = 40
            filled = int(bar_length * step / self.total_steps)
            bar = "=" * filled + "-" * (bar_length - filled)

            print(f"\r[{bar}] {progress:.1f}% Step {step}/{self.total_steps} | Speed: {speed:.2f} steps/s | ETA: {remaining:.0f}s", end="")

            if metrics:
                print(f"\nMetrics: {metrics}")

    def finish(self):
        """Finish tracking."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        print(f"\n{'='*60}")
        print(f"Training completed in {elapsed:.2f}s")
        print(f"Average speed: {self.total_steps / elapsed:.2f} steps/s")
        print("=" * 60)


# Import numpy for random state saving
try:
    import numpy as np
except ImportError:
    np = None
