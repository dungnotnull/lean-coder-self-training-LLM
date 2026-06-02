"""Knowledge Core: Gated improvement loop implementation."""

import sys
from pathlib import Path
from datetime import datetime
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from knowledge.processor import KnowledgeProcessor, QualityGate, ExampleCandidate
from knowledge.brain import Brain, BrainEntry
from registry.manager import ModelRegistry
from config import config


class ImprovementLoop:
    """Manages the gated improvement loop."""

    def __init__(self, brain_path: str, registry_path: str, eval_set_path: str):
        """Initialize improvement loop.

        Args:
            brain_path: Path to knowledge brain file
            registry_path: Path to model registry
            eval_set_path: Path to held-out eval set
        """
        self.brain = Brain(brain_path)
        self.registry = ModelRegistry(registry_path)
        self.quality_gate = QualityGate(
            check_license=True,
            check_correctness=True,
            check_relevance=True,
            check_dedup=True,
            check_leakage=True,
            check_quality=True,
        )
        self.processor = KnowledgeProcessor(self.quality_gate, eval_set_path)

    def add_examples(
        self,
        examples: list[ExampleCandidate],
        batch_name: str,
    ) -> tuple[int, int]:
        """Add examples through quality gate.

        Args:
            examples: List of candidate examples
            batch_name: Name for this batch

        Returns:
            Tuple of (accepted_count, rejected_count)
        """
        accepted = 0
        rejected = 0

        print(f"\nProcessing batch: {batch_name}")
        print(f"Total candidates: {len(examples)}")

        for example in examples:
            is_accepted, reason = self.processor.process_candidate(example)

            if is_accepted:
                accepted += 1
                print(f"✓ Accepted: {example.source}")
            else:
                rejected += 1
                self.processor.add_to_quarantine(example, reason)
                print(f"✗ Rejected: {example.source} - {reason}")

        print(f"\nResults: {accepted} accepted, {rejected} rejected")
        return accepted, rejected

    def create_dataset_version(
        self,
        version: str,
        base_version: str = None,
        added_examples: list = None,
    ) -> dict:
        """Create a new dataset version.

        Args:
            version: New version identifier (e.g., "ds-v2")
            base_version: Previous version to build on
            added_examples: Examples to add

        Returns:
            Version manifest
        """
        manifest = {
            "version": version,
            "base_version": base_version,
            "added_count": len(added_examples) if added_examples else 0,
            "created_at": datetime.now().isoformat(),
        }

        # Save manifest
        data_dir = Path(config.data_dir)
        version_path = data_dir / f"{version}_manifest.json"
        version_path.parent.mkdir(parents=True, exist_ok=True)

        with open(version_path, "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"\nDataset version {version} created")
        print(f"Base: {base_version}")
        print(f"Added examples: {manifest['added_count']}")

        return manifest

    def record_iteration(
        self,
        dataset_version: str,
        training_run: str,
        method: str,
        eval_results: dict,
        description: str = "",
    ) -> BrainEntry:
        """Record an improvement iteration.

        Args:
            dataset_version: Dataset version used
            training_run: Training run ID
            method: Training method (qlora-sft, dpo, etc.)
            eval_results: Evaluation results
            description: Description of changes

        Returns:
            BrainEntry
        """
        # Get best checkpoint for comparison
        best = self.registry.get_best()
        previous_best_pass = best.eval.pass_at_1 if best and best.eval else 0.0

        # Calculate delta
        current_pass = eval_results.get("pass_at_1", 0.0)
        vs_previous_best = current_pass - previous_best_pass

        # Determine decision
        promoted = current_pass > previous_best_pass if best else True
        decision = "promoted" if promoted else "rejected"

        # Create brain entry
        timestamp = datetime.now()
        entry_id = f"KB-{timestamp.strftime('%Y%m%d')}-{len(self.brain.entries) + 1:04d}"

        entry = BrainEntry(
            id=entry_id,
            dataset_version=dataset_version,
            added_examples=0,  # Would be filled from data manifest
            description=description or f"Iteration on {dataset_version}",
            training_run=training_run,
            method=method,
            eval=eval_results,
            vs_previous_best=vs_previous_best,
            decision=decision,
            brain_version=self.brain.current_version.version,
            created_at=timestamp,
        )

        self.brain.add_entry(entry)

        print(f"\n{'='*60}")
        print(f"Improvement Iteration Recorded")
        print(f"{'='*60}")
        print(f"Entry ID: {entry_id}")
        print(f"Dataset: {dataset_version}")
        print(f"Training run: {training_run}")
        print(f"Method: {method}")
        print(f"Pass@1: {current_pass:.2%}")
        print(f"vs previous best: {vs_previous_best:+.2%}")
        print(f"Decision: {decision.upper()}")
        print(f"Brain version: {entry.brain_version}")
        print(f"{'='*60}\n")

        return entry

    def get_quarantine_report(self) -> dict:
        """Get report of quarantined examples.

        Returns:
            Quarantine statistics
        """
        quarantine = self.processor.quarantine

        reasons = {}
        for item in quarantine:
            reason = item["reason"]
            reasons[reason] = reasons.get(reason, 0) + 1

        return {
            "total_quarantined": len(quarantine),
            "reasons": reasons,
        }


if __name__ == "__main__":
    # Example: Run an improvement iteration
    loop = ImprovementLoop(
        brain_path=str(Path(config.root_dir) / "knowledge" / "brain.json"),
        registry_path=str(config.registry_path),
        eval_set_path="data/eval/problems.json",
    )

    # Record a hypothetical iteration
    loop.record_iteration(
        dataset_version="ds-v2",
        training_run="qlora-20260602-140000",
        method="qlora-sft",
        eval_results={
            "pass_at_1": 0.42,
            "pass_at_k": [0.42, 0.60, 0.72],
            "latency_tok_s": 48.0,
            "size_mb": 14550,
        },
        description="Added 50 new coding examples from failure cases",
    )
