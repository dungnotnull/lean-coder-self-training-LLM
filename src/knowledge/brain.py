"""Brain version tracking for knowledge improvements."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
import json


@dataclass
class BrainEntry:
    """Entry in the knowledge improvement ledger."""

    id: str
    dataset_version: str
    added_examples: int
    description: str
    training_run: str
    method: str
    eval: dict
    vs_previous_best: float
    decision: str
    brain_version: str
    created_at: datetime

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "dataset_version": self.dataset_version,
            "added_examples": self.added_examples,
            "description": self.description,
            "training_run": self.training_run,
            "method": self.method,
            "eval": self.eval,
            "vs_previous_best": self.vs_previous_best,
            "decision": self.decision,
            "brain_version": self.brain_version,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class BrainVersion:
    """Current brain version state."""

    version: str
    best_checkpoint: Optional[str]
    dataset_version: Optional[str]
    iterations_promoted: int = 0
    iterations_rejected: int = 0


class Brain:
    """Knowledge brain for tracking improvements."""

    def __init__(self, brain_path: str):
        """Initialize brain.

        Args:
            brain_path: Path to brain JSON file
        """
        self.brain_path = Path(brain_path)
        self.brain_path.parent.mkdir(parents=True, exist_ok=True)

        self.entries: list[BrainEntry] = []
        self.current_version = BrainVersion(
            version="v1.0",
            best_checkpoint=None,
            dataset_version=None,
        )

        self._load()

    def _load(self):
        """Load existing brain or create new."""
        if self.brain_path.exists():
            with open(self.brain_path) as f:
                data = json.load(f)

            for entry_data in data.get("improvement_ledger", []):
                self.entries.append(
                    BrainEntry(
                        id=entry_data["id"],
                        dataset_version=entry_data["dataset_version"],
                        added_examples=entry_data["added_examples"],
                        description=entry_data["description"],
                        training_run=entry_data["training_run"],
                        method=entry_data["method"],
                        eval=entry_data["eval"],
                        vs_previous_best=entry_data["vs_previous_best"],
                        decision=entry_data["decision"],
                        brain_version=entry_data["brain_version"],
                        created_at=datetime.fromisoformat(entry_data["created_at"]),
                    )
                )

            version_data = data.get("versioning", {})
            self.current_version = BrainVersion(
                version=version_data.get("current", "v1.0"),
                best_checkpoint=version_data.get("current_best_checkpoint"),
                dataset_version=version_data.get("current_dataset_version"),
                iterations_promoted=version_data.get("iterations", 0),
            )

    def _save(self):
        """Save brain to disk."""
        data = {
            "improvement_ledger": [e.to_dict() for e in self.entries],
            "versioning": {
                "current": self.current_version.version,
                "current_best_checkpoint": self.current_version.best_checkpoint,
                "current_dataset_version": self.current_version.dataset_version,
                "iterations": self.current_version.iterations_promoted,
            },
        }

        with open(self.brain_path, "w") as f:
            json.dump(data, f, indent=2)

    def add_entry(self, entry: BrainEntry):
        """Add entry to brain ledger.

        Args:
            entry: BrainEntry to add
        """
        self.entries.append(entry)

        if entry.decision == "promoted":
            self.current_version.iterations_promoted += 1
            # Bump version
            major, minor = map(int, self.current_version.version[1:].split("."))
            minor += 1
            self.current_version.version = f"v{major}.{minor}"
            self.current_version.best_checkpoint = entry.training_run
            self.current_version.dataset_version = entry.dataset_version
        else:
            self.current_version.iterations_rejected += 1

        self._save()
