"""Data build runner script."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.pipeline import DataPipeline
from config import config


def build_dataset_version(
    version: str,
    sources: list[dict],
    eval_set_path: str = None,
):
    """Build a dataset version.

    Args:
        version: Dataset version (e.g., "ds-v1")
        sources: List of source configurations
        eval_set_path: Optional path to eval set
    """
    data_config = config.load_yaml("config/data.yaml")

    pipeline = DataPipeline(data_config)

    stats = pipeline.build_version(
        version=version,
        sources=sources,
        eval_set_path=eval_set_path,
    )

    print(f"\n{'='*50}")
    print(f"Dataset {version} Summary:")
    print(f"{'='*50}")
    print(f"Total examples: {stats.total_examples}")
    print(f"Train: {stats.train_size}")
    print(f"Val: {stats.val_size}")
    print(f"Holdout: {stats.holdout_size}")
    print(f"Exact duplicates removed: {stats.dedup_exact_removed}")
    print(f"Near duplicates removed: {stats.dedup_near_removed}")
    print(f"Leakage checked: {stats.leakage_checked}")
    print(f"{'='*50}")


if __name__ == "__main__":
    # Build ds-v1 from sample data
    sources = [
        {
            "name": "sample_coding_qa",
            "path": "data/train/ds-v1_train.jsonl",
            "type": "jsonl",
            "license": "apache-2.0",
        }
    ]

    build_dataset_version(
        version="ds-v1",
        sources=sources,
        eval_set_path="data/eval/problems.json",
    )
