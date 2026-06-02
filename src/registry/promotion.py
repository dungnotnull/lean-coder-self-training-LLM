"""Promotion script for comparing and promoting checkpoints."""

import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from registry.manager import ModelRegistry, CheckpointInfo, EvalResult


def compare_and_promote(
    checkpoint_id: str,
    registry_path: str,
    primary_metric: str = "pass_at_1",
) -> dict:
    """Compare checkpoint against best and promote if better.

    Args:
        checkpoint_id: Checkpoint to evaluate
        registry_path: Path to registry
        primary_metric: Metric for comparison

    Returns:
        Comparison results
    """
    registry = ModelRegistry(registry_path)

    checkpoint = registry.get_checkpoint(checkpoint_id)
    if not checkpoint:
        raise ValueError(f"Checkpoint not found: {checkpoint_id}")

    if not checkpoint.eval:
        raise ValueError(f"Checkpoint has no eval results: {checkpoint_id}")

    best = registry.get_best()

    results = {
        "checkpoint_id": checkpoint_id,
        "primary_metric": primary_metric,
        "checkpoint_value": getattr(checkpoint.eval, primary_metric, 0),
        "best_id": best.id if best else None,
        "best_value": getattr(best.eval, primary_metric, 0) if best else None,
        "promoted": False,
    }

    if best:
        delta = results["checkpoint_value"] - results["best_value"]
        results["delta"] = delta
        print(f"\n{'='*60}")
        print(f"Comparison: {checkpoint_id} vs {best.id}")
        print(f"{'='*60}")
        print(f"Primary metric: {primary_metric}")
        print(f"{checkpoint_id}: {results['checkpoint_value']:.2%}")
        print(f"{best.id}: {results['best_value']:.2%}")
        print(f"Delta: {delta:+.2%}")
    else:
        print(f"\n{'='*60}")
        print(f"No current best checkpoint found")
        print(f"{'='*60}")
        print(f"{checkpoint_id}: {results['checkpoint_value']:.2%}")

    # Try to promote
    promoted = registry.promote_if_better(checkpoint_id, primary_metric)

    results["promoted"] = promoted
    results["new_best"] = promoted

    if promoted:
        print(f"\n✓ PROMOTED to best!")
        print(f"New best checkpoint: {checkpoint_id}")
    elif best:
        print(f"\n✗ NOT promoted (does not beat current best)")
    else:
        print(f"\n✓ Auto-promoted (first checkpoint with eval results)")

    # Also compare to base baseline
    all_checkpoints = registry.list_checkpoints()
    base_checkpoint = next((cp for cp in all_checkpoints if cp.source == "base"), None)

    if base_checkpoint and base_checkpoint.id != checkpoint_id:
        base_delta = results["checkpoint_value"] - getattr(base_checkpoint.eval, primary_metric, 0)
        results["vs_base"] = base_delta
        print(f"\nvs base baseline: {base_delta:+.2%}")

    print(f"{'='*60}\n")

    return results


def generate_comparison_report(checkpoint_id: str, registry_path: str) -> dict:
    """Generate full comparison report.

    Args:
        checkpoint_id: Checkpoint to report on
        registry_path: Path to registry

    Returns:
        Full comparison dict
    """
    registry = ModelRegistry(registry_path)
    checkpoint = registry.get_checkpoint(checkpoint_id)

    if not checkpoint or not checkpoint.eval:
        return {"error": "Checkpoint or eval results not found"}

    best = registry.get_best()
    all_checkpoints = registry.list_checkpoints()

    comparison = {
        "checkpoint": {
            "id": checkpoint.id,
            "source": checkpoint.source,
            "dataset_version": checkpoint.dataset_version,
            "pass_at_1": checkpoint.eval.pass_at_1,
            "pass_at_10": checkpoint.eval.pass_at_k[2] if len(checkpoint.eval.pass_at_k) > 2 else 0,
            "size_mb": checkpoint.eval.size_mb,
            "latency_tok_s": checkpoint.eval.latency_tok_s,
        },
        "vs_best": {},
        "vs_base": {},
    }

    if best:
        comparison["vs_best"] = {
            "id": best.id,
            "pass_at_1_delta": checkpoint.eval.pass_at_1 - best.eval.pass_at_1,
            "pass_at_10_delta": (checkpoint.eval.pass_at_k[2] - best.eval.pass_at_k[2]) if len(checkpoint.eval.pass_at_k) > 2 else 0,
            "size_mb_delta": checkpoint.eval.size_mb - best.eval.size_mb,
        }

    base_checkpoint = next((cp for cp in all_checkpoints if cp.source == "base"), None)
    if base_checkpoint:
        comparison["vs_base"] = {
            "id": base_checkpoint.id,
            "pass_at_1_delta": checkpoint.eval.pass_at_1 - base_checkpoint.eval.pass_at_1,
            "pass_at_10_delta": (checkpoint.eval.pass_at_k[2] - base_checkpoint.eval.pass_at_k[2]) if len(base_checkpoint.eval.pass_at_k) > 2 else 0,
            "size_mb_delta": checkpoint.eval.size_mb - base_checkpoint.eval.size_mb,
        }

    return comparison


if __name__ == "__main__":
    # Example: Compare and promote a checkpoint
    result = compare_and_promote(
        checkpoint_id="qlora-20260602-120000",
        registry_path=str(Path(__file__).parent.parent.parent / "registry" / "models.json"),
    )
