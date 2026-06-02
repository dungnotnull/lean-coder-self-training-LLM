"""Data pipeline for building versioned datasets."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import json
from datetime import datetime

from .builder import DatasetBuilder, DatasetRecipe
from .cleaners import DataCleaner
from .dedup import Deduplicator
from .formatter import ChatFormatter


@dataclass
class DataStats:
    """Statistics for a dataset."""

    total_examples: int
    train_size: int
    val_size: int
    holdout_size: int
    sources: list[dict]
    dedup_exact_removed: int = 0
    dedup_near_removed: int = 0
    leakage_checked: bool = False


class DataPipeline:
    """Complete data pipeline for versioned datasets."""

    def __init__(self, config: dict):
        """Initialize data pipeline.

        Args:
            config: Data pipeline configuration
        """
        self.config = config
        self.output_dir = Path(config.get("output_dir", "data"))
        self.builder = DatasetBuilder(config)
        self.cleaner = DataCleaner(config.get("cleaning", {}))
        self.dedup = Deduplicator(config.get("dedup", {}))
        self.formatter = ChatFormatter(config.get("formatting", {}))

    def build_version(
        self,
        version: str,
        sources: list[dict],
        eval_set_path: Optional[str] = None,
    ) -> DataStats:
        """Build a versioned dataset.

        Args:
            version: Dataset version (e.g., "ds-v1")
            sources: List of data sources with metadata
            eval_set_path: Optional path to eval set for leakage check

        Returns:
            DataStats with build statistics
        """
        print(f"Building dataset version: {version}")

        # Step 1: Collect data from sources
        all_data = []
        for source in sources:
            source_data = self._load_source(source)
            all_data.extend(source_data)
            print(f"Loaded {len(source_data)} examples from {source['name']}")

        print(f"Total raw examples: {len(all_data)}")

        # Step 2: Clean data
        print("Cleaning data...")
        cleaned_data = self.cleaner.clean(all_data)
        print(f"After cleaning: {len(cleaned_data)} examples")

        # Step 3: Deduplicate
        print("Deduplicating (exact)...")
        deduped_data = self.dedup.dedup_exact(cleaned_data, key="text")
        exact_removed = len(all_data) - len(deduped_data)
        print(f"Removed {exact_removed} exact duplicates")

        print("Deduplicating (near)...")
        near_deduped = self.dedup.dedup_near(deduped_data)
        near_removed = len(deduped_data) - len(near_deduped)
        print(f"Removed {near_removed} near duplicates")

        # Step 4: Check for eval leakage
        leakage_checked = False
        if eval_set_path:
            print("Checking for eval leakage...")
            eval_data = self._load_source({"path": eval_set_path, "type": "jsonl"})
            leakage_checked = self.dedup.dedup_vs_eval(near_deduped, eval_data)
            print(f"Leakage check passed: {leakage_checked}")

        # Step 5: Split into train/val/holdout
        print("Splitting data...")
        train_data, val_data, holdout_data = self._split_data(near_deduped)

        # Step 6: Format and save
        print("Formatting and saving...")
        self._save_dataset(train_data, self.output_dir / "train" / f"{version}_train.jsonl")
        self._save_dataset(val_data, self.output_dir / "val" / f"{version}_val.jsonl")
        self._save_dataset(holdout_data, self.output_dir / "holdout" / f"{version}_holdout.jsonl")

        # Step 7: Save manifest
        stats = DataStats(
            total_examples=len(near_deduped),
            train_size=len(train_data),
            val_size=len(val_data),
            holdout_size=len(holdout_data),
            sources=sources,
            dedup_exact_removed=exact_removed,
            dedup_near_removed=near_removed,
            leakage_checked=leakage_checked,
        )

        self._save_manifest(version, stats)

        print(f"Dataset {version} built successfully!")
        return stats

    def _load_source(self, source: dict) -> list[dict]:
        """Load data from a source.

        Args:
            source: Source configuration

        Returns:
            List of examples
        """
        source_path = Path(source.get("path", ""))
        source_type = source.get("type", "jsonl")

        if not source_path.exists():
            print(f"Source path not found: {source_path}")
            return []

        examples = []
        if source_type == "jsonl":
            with open(source_path) as f:
                for line in f:
                    if line.strip():
                        examples.append(json.loads(line))
        elif source_type == "json":
            with open(source_path) as f:
                data = json.load(f)
                examples = data if isinstance(data, list) else data.get("examples", [])

        return examples

    def _split_data(
        self, data: list[dict], train_ratio: float = 0.8, val_ratio: float = 0.1
    ) -> tuple[list[dict], list[dict], list[dict]]:
        """Split data into train/val/holdout.

        Args:
            data: List of examples
            train_ratio: Ratio for training
            val_ratio: Ratio for validation

        Returns:
            Tuple of (train, val, holdout)
        """
        n = len(data)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        return data[:train_end], data[train_end:val_end], data[val_end:]

    def _save_dataset(self, data: list[dict], path: Path):
        """Save dataset to file.

        Args:
            data: List of examples
            path: Output path
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for example in data:
                f.write(json.dumps(example) + "\n")

    def _save_manifest(self, version: str, stats: DataStats):
        """Save dataset manifest.

        Args:
            version: Dataset version
            stats: Data statistics
        """
        manifest_path = self.output_dir / f"{version}_manifest.json"

        manifest = {
            "version": version,
            "created_at": datetime.now().isoformat(),
            "stats": {
                "total_examples": stats.total_examples,
                "train_size": stats.train_size,
                "val_size": stats.val_size,
                "holdout_size": stats.holdout_size,
                "dedup_exact_removed": stats.dedup_exact_removed,
                "dedup_near_removed": stats.dedup_near_removed,
                "leakage_checked": stats.leakage_checked,
            },
            "sources": stats.sources,
        }

        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
