"""Model checkpoint registry with version tracking and promotion logic."""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional
import json


@dataclass
class EvalResult:
    """Evaluation metrics for a checkpoint."""

    pass_at_1: float
    pass_at_k: list[float]
    latency_tok_s: float
    size_mb: float
    memory_mb: float

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class CheckpointInfo:
    """Registry entry for a checkpoint."""

    id: str
    source: str
    base_model: str
    dataset_version: Optional[str]
    path: str
    eval: Optional[EvalResult]
    is_best: bool
    created_at: datetime
    manifest: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "source": self.source,
            "base_model": self.base_model,
            "dataset_version": self.dataset_version,
            "path": self.path,
            "eval": self.eval.to_dict() if self.eval else None,
            "is_best": self.is_best,
            "created_at": self.created_at.isoformat(),
            "manifest": self.manifest,
        }


class ModelRegistry:
    """Manages versioned checkpoint registry."""

    def __init__(self, registry_path: str):
        """Load or create registry file.

        Args:
            registry_path: Path to registry JSON file
        """
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)

        self.checkpoints: dict[str, CheckpointInfo] = {}
        self.best_pointer: Optional[str] = None

        self._load()

    def _load(self):
        """Load existing registry or create new."""
        if self.registry_path.exists() and self.registry_path.stat().st_size > 0:
            with open(self.registry_path) as f:
                data = json.load(f)

            for cp_data in data.get("checkpoints", []):
                eval_data = cp_data.get("eval")
                eval_result = (
                    EvalResult(**eval_data) if eval_data else None
                )

                self.checkpoints[cp_data["id"]] = CheckpointInfo(
                    id=cp_data["id"],
                    source=cp_data["source"],
                    base_model=cp_data["base_model"],
                    dataset_version=cp_data.get("dataset_version"),
                    path=cp_data["path"],
                    eval=eval_result,
                    is_best=cp_data["is_best"],
                    created_at=datetime.fromisoformat(cp_data["created_at"]),
                    manifest=cp_data.get("manifest", {}),
                )

            self.best_pointer = data.get("best_pointer")
        else:
            # Initialize empty registry
            self._save()

    def _save(self):
        """Save registry to disk."""
        data = {
            "checkpoints": [cp.to_dict() for cp in self.checkpoints.values()],
            "best_pointer": self.best_pointer,
        }

        with open(self.registry_path, "w") as f:
            json.dump(data, f, indent=2)

    def register_checkpoint(self, info: CheckpointInfo) -> str:
        """Add a new checkpoint to registry.

        Args:
            info: CheckpointInfo to register

        Returns:
            Checkpoint ID
        """
        self.checkpoints[info.id] = info
        self._save()
        return info.id

    def get_checkpoint(self, checkpoint_id: str) -> Optional[CheckpointInfo]:
        """Get checkpoint by ID.

        Args:
            checkpoint_id: Checkpoint identifier

        Returns:
            CheckpointInfo if found, None otherwise
        """
        return self.checkpoints.get(checkpoint_id)

    def get_best(self) -> Optional[CheckpointInfo]:
        """Get the current best checkpoint.

        Returns:
            CheckpointInfo for best checkpoint, None if no checkpoints
        """
        if self.best_pointer and self.best_pointer in self.checkpoints:
            return self.checkpoints[self.best_pointer]

        # Fallback: find first checkpoint with is_best=True
        for cp in self.checkpoints.values():
            if cp.is_best:
                self.best_pointer = cp.id
                self._save()
                return cp

        return None

    def promote_if_better(
        self, new_id: str, primary_metric: str = "pass_at_1"
    ) -> bool:
        """Promote new checkpoint to "best" if it beats current best.

        Args:
            new_id: Checkpoint ID to evaluate
            primary_metric: Metric to compare (default: pass_at_1)

        Returns:
            True if promoted, False otherwise
        """
        new_cp = self.get_checkpoint(new_id)
        if not new_cp or not new_cp.eval:
            return False

        best_cp = self.get_best()

        # No current best? Auto-promote if has eval results
        if not best_cp:
            new_cp.is_best = True
            self.best_pointer = new_id
            for cp in self.checkpoints.values():
                if cp.id != new_id:
                    cp.is_best = False
            self._save()
            return True

        # Compare metrics
        new_value = getattr(new_cp.eval, primary_metric, 0)
        best_value = getattr(best_cp.eval, primary_metric, 0)

        if new_value > best_value:
            # Promote new
            new_cp.is_best = True
            best_cp.is_best = False
            self.best_pointer = new_id
            self._save()
            return True

        return False

    def list_checkpoints(self) -> list[CheckpointInfo]:
        """List all checkpoints.

        Returns:
            List of CheckpointInfo sorted by creation time (newest first)
        """
        return sorted(
            self.checkpoints.values(), key=lambda x: x.created_at, reverse=True
        )
