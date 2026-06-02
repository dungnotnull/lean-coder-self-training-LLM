"""Registry view for dashboard."""

from typing import List, Dict, Optional
from datetime import datetime


class RegistryView:
    """Registry data for dashboard display."""

    def __init__(self, registry_path: str):
        """Initialize registry view.

        Args:
            registry_path: Path to model registry
        """
        self.registry_path = registry_path
        self._registry = None

    def _load_registry(self):
        """Lazy load registry."""
        if self._registry is None:
            from registry.manager import ModelRegistry
            self._registry = ModelRegistry(self.registry_path)

    def get_summary(self) -> Dict:
        """Get registry summary.

        Returns:
            Summary dictionary
        """
        self._load_registry()

        checkpoints = self._registry.list_checkpoints()
        best = self._registry.get_best()

        return {
            "total_checkpoints": len(checkpoints),
            "best_checkpoint": best.id if best else None,
            "best_score": best.eval.pass_at_1 if best and best.eval else 0.0,
            "recent_checkpoints": [
                {
                    "id": cp.id,
                    "source": cp.source,
                    "pass_at_1": cp.eval.pass_at_1 if cp.eval else 0.0,
                    "created_at": cp.created_at.isoformat(),
                }
                for cp in checkpoints[:5]
            ],
        }

    def get_checkpoint(self, checkpoint_id: str) -> Dict:
        """Get checkpoint details.

        Args:
            checkpoint_id: Checkpoint ID

        Returns:
            Checkpoint details
        """
        self._load_registry()
        checkpoint = self._registry.get_checkpoint(checkpoint_id)

        if not checkpoint:
            return {"error": "Checkpoint not found"}

        return {
            "id": checkpoint.id,
            "source": checkpoint.source,
            "base_model": checkpoint.base_model,
            "dataset_version": checkpoint.dataset_version,
            "path": checkpoint.path,
            "is_best": checkpoint.is_best,
            "created_at": checkpoint.created_at.isoformat(),
            "eval": {
                "pass_at_1": checkpoint.eval.pass_at_1 if checkpoint.eval else 0.0,
                "pass_at_10": checkpoint.eval.pass_at_k[2] if checkpoint.eval and len(checkpoint.eval.pass_at_k) > 2 else 0.0,
                "latency_tok_s": checkpoint.eval.latency_tok_s if checkpoint.eval else 0.0,
                "size_mb": checkpoint.eval.size_mb if checkpoint.eval else 0.0,
            } if checkpoint.eval else None,
            "manifest": checkpoint.manifest,
        }

    def get_history(self) -> List[Dict]:
        """Get checkpoint history.

        Returns:
            List of checkpoints with scores
        """
        self._load_registry()
        checkpoints = self._registry.list_checkpoints()

        history = []
        for cp in checkpoints:
            history.append({
                "id": cp.id,
                "source": cp.source,
                "is_best": cp.is_best,
                "pass_at_1": cp.eval.pass_at_1 if cp.eval else 0.0,
                "size_mb": cp.eval.size_mb if cp.eval else 0.0,
                "created_at": cp.created_at.isoformat(),
            })

        return history


class KnowledgeBrainView:
    """Knowledge brain data for dashboard."""

    def __init__(self, brain_path: str):
        """Initialize brain view.

        Args:
            brain_path: Path to knowledge brain
        """
        self.brain_path = brain_path
        self._brain = None

    def _load_brain(self):
        """Lazy load brain."""
        if self._brain is None:
            from knowledge.brain import Brain
            self._brain = Brain(self.brain_path)

    def get_summary(self) -> Dict:
        """Get brain summary.

        Returns:
            Summary dictionary
        """
        self._load_brain()

        return {
            "current_version": self._brain.current_version.version,
            "best_checkpoint": self._brain.current_version.best_checkpoint,
            "dataset_version": self._brain.current_version.dataset_version,
            "iterations_promoted": self._brain.current_version.iterations_promoted,
            "iterations_rejected": self._brain.current_version.iterations_rejected,
            "total_entries": len(self._brain.entries),
        }

    def get_timeline(self) -> List[Dict]:
        """Get improvement timeline.

        Returns:
            List of brain entries
        """
        self._load_brain()

        timeline = []
        for entry in self._brain.entries:
            timeline.append({
                "id": entry.id,
                "dataset_version": entry.dataset_version,
                "method": entry.method,
                "vs_previous_best": entry.vs_previous_best,
                "decision": entry.decision,
                "brain_version": entry.brain_version,
                "created_at": entry.created_at.isoformat(),
            })

        return timeline
