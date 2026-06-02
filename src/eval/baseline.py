"""Baseline evaluation runner."""

from datetime import datetime
from pathlib import Path
from typing import Optional
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from base.fetch import download_base_model
from registry.manager import ModelRegistry, CheckpointInfo, EvalResult
from eval.harness import EvalHarness
from eval.reporter import EvalReporter


def run_baseline(
    model_config: dict,
    eval_config: dict,
    registry_path: str,
    hf_token: Optional[str] = None,
) -> CheckpointInfo:
    """Run complete baseline: download → eval → record → promote.

    Args:
        model_config: Base model config
        eval_config: Evaluation config
        registry_path: Path to registry file
        hf_token: Optional HF token

    Returns:
        Registered baseline checkpoint info
    """
    # Step 1: Download base model
    print("Downloading base model...")
    model_info = download_base_model(model_config, hf_token)
    print(f"Downloaded: {model_info.repo_id} ({model_info.size_mb:.0f} MB)")

    # Step 2: Create checkpoint info
    checkpoint_id = f"base-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # Step 3: Load model for eval (placeholder - would load actual model)
    print("Loading model for evaluation...")
    # model = load_model(model_info.path)  # Placeholder
    model = None  # For now, skip actual model loading

    # Step 4: Run eval
    print("Running evaluation...")
    harness = EvalHarness(eval_config)
    problems = harness.load_problems(eval_config.get("problems", {}).get("path", ""))

    n_samples = eval_config.get("metrics", {}).get("n_samples", [10])[0]
    eval_result = harness.run_eval(model, problems, n_samples, suite_name="baseline")

    # Override with placeholder size since we have the actual model info
    eval_result.model_size_mb = model_info.size_mb

    print(f"Eval complete: Pass@1 = {eval_result.pass_at_1:.2%}")

    # Step 5: Write report
    reporter = EvalReporter(eval_config.get("reporting", {}).get("outputs_dir", "outputs/eval"))
    report_path = reporter.write_report(eval_result, checkpoint_id)
    print(f"Report written to: {report_path}")

    # Step 6: Register checkpoint
    registry = ModelRegistry(registry_path)

    checkpoint_info = CheckpointInfo(
        id=checkpoint_id,
        source="base",
        base_model=model_info.repo_id,
        dataset_version=None,
        path=str(model_info.path),
        eval=EvalResult(
            pass_at_1=eval_result.pass_at_1,
            pass_at_k=eval_result.pass_at_k,
            latency_tok_s=eval_result.latency_tok_s,
            size_mb=eval_result.model_size_mb,
            memory_mb=eval_result.memory_mb,
        ),
        is_best=True,  # First checkpoint is best by default
        created_at=datetime.now(),
        manifest={
            "model_info": model_info.to_dict(),
            "eval_config": eval_config,
        },
    )

    registry.register_checkpoint(checkpoint_info)
    registry.promote_if_better(checkpoint_id)

    print(f"Checkpoint registered: {checkpoint_id}")
    print(f"Current best: {registry.best_pointer}")

    return checkpoint_info


if __name__ == "__main__":
    # Quick test
    from src.config import config

    model_cfg = config.load_yaml("config/base.yaml")
    eval_cfg = config.load_yaml("config/eval.yaml")

    run_baseline(
        model_config=model_cfg,
        eval_config=eval_cfg,
        registry_path=str(config.registry_path),
        hf_token=config.hf_token,
    )
