"""Model ensemble for weighted voting from top checkpoints."""

from typing import List, Dict, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class EnsembleMember:
    """A member of the ensemble."""

    checkpoint_id: str
    model_path: str
    weight: float
    eval_score: float


class ModelEnsemble:
    """Ensemble of multiple model checkpoints."""

    def __init__(self, members: List[EnsembleMember] = None):
        """Initialize model ensemble.

        Args:
            members: List of ensemble members
        """
        self.members = members or []
        self.models = {}

    def add_member(
        self,
        checkpoint_id: str,
        model_path: str,
        weight: float,
        eval_score: float,
    ):
        """Add a member to ensemble.

        Args:
            checkpoint_id: Checkpoint identifier
            model_path: Path to model
            weight: Weight for voting
            eval_score: Evaluation score
        """
        self.members.append(
            EnsembleMember(
                checkpoint_id=checkpoint_id,
                model_path=model_path,
                weight=weight,
                eval_score=eval_score,
            )
        )

    def load_models(self):
        """Load all models in ensemble."""
        # Placeholder: would load actual models
        for member in self.members:
            self.models[member.checkpoint_id] = f"Model({member.checkpoint_id})"
            print(f"Loaded {member.checkpoint_id}")

    def predict(
        self,
        prompt: str,
        method: str = "weighted_vote",
        n_tokens: int = 512,
    ) -> str:
        """Generate prediction using ensemble.

        Args:
            prompt: Input prompt
            method: Aggregation method (weighted_vote, majority, best_only)
            n_tokens: Max tokens to generate

        Returns:
            Generated response
        """
        if not self.members:
            raise ValueError("No ensemble members")

        if method == "best_only":
            # Use only best model
            best = max(self.members, key=lambda m: m.eval_score)
            return self._generate(best, prompt, n_tokens)

        elif method == "weighted_vote":
            # Weighted voting
            responses = []
            weights = []

            for member in self.members:
                response = self._generate(member, prompt, n_tokens)
                responses.append(response)
                weights.append(member.weight)

            # Return weighted response (simplified: highest weight)
            best_idx = np.argmax(weights)
            return responses[best_idx]

        elif method == "majority":
            # Majority voting
            responses = []
            for member in self.members:
                response = self._generate(member, prompt, n_tokens)
                responses.append(response)

            # Return most common response
            return max(set(responses), key=responses.count)

        else:
            raise ValueError(f"Unknown method: {method}")

    def _generate(self, member: EnsembleMember, prompt: str, n_tokens: int) -> str:
        """Generate from a single member.

        Args:
            member: Ensemble member
            prompt: Input prompt
            n_tokens: Max tokens

        Returns:
            Generated response
        """
        # Placeholder: would call actual model
        return f"Response from {member.checkpoint_id}"

    def create_from_registry(
        self,
        registry_path: str,
        top_n: int = 3,
        weight_strategy: str = "uniform",
    ) -> "ModelEnsemble":
        """Create ensemble from top N checkpoints.

        Args:
            registry_path: Path to registry
            top_n: Number of top checkpoints to use
            weight_strategy: Strategy for weights (uniform, score_based, rank_based)

        Returns:
            ModelEnsemble instance
        """
        from registry.manager import ModelRegistry

        registry = ModelRegistry(registry_path)
        checkpoints = registry.list_checkpoints()[:top_n]

        ensemble = ModelEnsemble()

        for i, cp in enumerate(checkpoints):
            if weight_strategy == "uniform":
                weight = 1.0 / len(checkpoints)
            elif weight_strategy == "score_based":
                score = cp.eval.pass_at_1 if cp.eval else 0.0
                weight = score / sum(c.eval.pass_at_1 for c in checkpoints if c.eval)
            elif weight_strategy == "rank_based":
                weight = 1.0 / (i + 1)
            else:
                weight = 1.0

            ensemble.add_member(
                checkpoint_id=cp.id,
                model_path=cp.path,
                weight=weight,
                eval_score=cp.eval.pass_at_1 if cp.eval else 0.0,
            )

        return ensemble

    def evaluate_ensemble(
        self,
        eval_problems: List,
        method: str = "weighted_vote",
    ) -> Dict:
        """Evaluate ensemble on problems.

        Args:
            eval_problems: List of evaluation problems
            method: Aggregation method

        Returns:
            Evaluation results
        """
        results = {"total": len(eval_problems), "passed": 0, "failed": 0}

        for problem in eval_problems:
            response = self.predict(
                problem.get("prompt", ""),
                method=method,
            )

            # Test response (placeholder)
            passed = self._test_response(response, problem)
            if passed:
                results["passed"] += 1
            else:
                results["failed"] += 1

        results["pass_rate"] = results["passed"] / results["total"] if results["total"] > 0 else 0.0

        return results

    def _test_response(self, response: str, problem: Dict) -> bool:
        """Test if response solves problem.

        Args:
            response: Generated response
            problem: Problem with tests

        Returns:
            True if passes tests
        """
        # Placeholder: would execute code against tests
        return bool(response and len(response) > 10)

    def __len__(self) -> int:
        """Get number of members."""
        return len(self.members)

    def __repr__(self) -> str:
        """String representation."""
        return f"ModelEnsemble({len(self.members)} members)"
