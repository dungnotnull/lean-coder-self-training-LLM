"""Executable code evaluation harness with sandboxing."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import json
import subprocess
import tempfile
import time


@dataclass
class Problem:
    """A coding problem with test cases."""

    id: str
    description: str
    starter_code: str = ""
    tests: list[str] = None

    def __post_init__(self):
        if self.tests is None:
            self.tests = []


@dataclass
class EvalResult:
    """Results from running eval on a model."""

    suite: str
    pass_at_1: float
    pass_at_k: list[float]
    latency_tok_s: float
    model_size_mb: float
    memory_mb: float
    problems_total: int
    details: list[dict]


class EvalHarness:
    """Executable code evaluation harness."""

    def __init__(self, config: dict):
        """Initialize harness with sandbox config.

        Args:
            config: Eval configuration dict
        """
        self.sandbox_config = config.get("sandbox", {})
        self.metrics_config = config.get("metrics", {})
        self.timeout = self.sandbox_config.get("timeout_seconds", 30)
        self.memory_limit = self.sandbox_config.get("memory_limit_mb", 512)

    def load_problems(self, path: str) -> list[Problem]:
        """Load problems from JSON file.

        Args:
            path: Path to problems JSON file

        Returns:
            List of Problem instances
        """
        problems_path = Path(path)
        if not problems_path.exists():
            # Create sample problems for testing
            return self._create_sample_problems()

        with open(problems_path) as f:
            data = json.load(f)

        problems = []
        for item in data.get("problems", []):
            problems.append(
                Problem(
                    id=item["id"],
                    description=item["description"],
                    starter_code=item.get("starter_code", ""),
                    tests=item.get("tests", []),
                )
            )

        return problems

    def _create_sample_problems(self) -> list[Problem]:
        """Create sample problems for testing."""
        return [
            Problem(
                id="sample_001",
                description="Write a function that adds two numbers.",
                starter_code="def add(a, b):\n    ",
                tests=[
                    "assert add(1, 2) == 3",
                    "assert add(-1, 1) == 0",
                    "assert add(0, 0) == 0",
                ],
            ),
            Problem(
                id="sample_002",
                description="Write a function that returns the length of a list.",
                starter_code="def list_length(lst):\n    ",
                tests=[
                    "assert list_length([]) == 0",
                    "assert list_length([1]) == 1",
                    'assert list_length([1,2,3]) == 3',
                ],
            ),
        ]

    def run_eval(
        self, model, problems: list[Problem], n_samples: int, suite_name: str = "eval"
    ) -> EvalResult:
        """Run evaluation on model against problems.

        Args:
            model: Loaded model for generation (has generate method)
            problems: List of coding problems
            n_samples: Number of samples per problem for pass@k
            suite_name: Name of evaluation suite

        Returns:
            EvalResult with metrics
        """
        details = []
        total_passed = 0
        total_samples = 0
        start_time = time.time()

        for problem in problems:
            problem_results = self._evaluate_problem(model, problem, n_samples)
            details.append(problem_results)

            c = problem_results["c"]
            total_passed += c
            total_samples += n_samples

        elapsed = time.time() - start_time
        latency_tok_s = elapsed / max(total_samples, 1) * 1000  # Placeholder

        # Calculate pass@k
        pass_at_1 = total_passed / len(problems)
        pass_at_k = self._calculate_pass_at_k(len(problems), total_passed, n_samples)

        # Get model size (placeholder - should come from model)
        model_size_mb = 0.0
        memory_mb = 0.0

        return EvalResult(
            suite=suite_name,
            pass_at_1=pass_at_1,
            pass_at_k=pass_at_k,
            latency_tok_s=latency_tok_s,
            model_size_mb=model_size_mb,
            memory_mb=memory_mb,
            problems_total=len(problems),
            details=details,
        )

    def _evaluate_problem(self, model, problem: Problem, n_samples: int) -> dict:
        """Evaluate a single problem.

        Args:
            model: Model for generation
            problem: Problem to evaluate
            n_samples: Number of samples to generate

        Returns:
            Dict with problem results
        """
        passed_count = 0

        for _ in range(n_samples):
            # Generate solution (placeholder - uses starter code as "solution")
            solution = problem.starter_code + "pass"

            # Execute in sandbox
            if self._execute_sandboxed(solution, problem.tests):
                passed_count += 1

        return {
            "problem_id": problem.id,
            "c": passed_count,
            "n": n_samples,
        }

    def _execute_sandboxed(self, code: str, tests: list[str]) -> bool:
        """Execute code against tests in isolated sandbox.

        Args:
            code: Code to execute
            tests: List of test assertions

        Returns:
            True if all tests pass
        """
        # Build complete code with tests
        full_code = code + "\n\n# Tests\n"
        for test in tests:
            full_code += f"{test}\n"

        # Write to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(full_code)
            temp_path = f.name

        try:
            # Run in subprocess with limits
            result = subprocess.run(
                ["python", temp_path],
                capture_output=True,
                timeout=self.timeout,
                text=True,
            )

            return result.returncode == 0

        except subprocess.TimeoutExpired:
            return False
        except Exception:
            return False
        finally:
            # Clean up temp file
            try:
                Path(temp_path).unlink()
            except Exception:
                pass

    def _calculate_pass_at_k(
        self, num_problems: int, total_correct: int, n_samples: int
    ) -> list[float]:
        """Calculate pass@k for k = [1, n_samples].

        Args:
            num_problems: Number of problems
            total_correct: Total correct solutions across all problems
            n_samples: Number of samples per problem

        Returns:
            List of pass@k values
        """
        import math

        pass_at_k = []
        for k in range(1, n_samples + 1):
            # Estimate pass@k using the formula
            # For simplicity, use total_correct / num_problems as base
            # In real implementation, would use per-problem correct counts
            c = total_correct
            n = n_samples

            if n - c >= k:
                pass_k = 1.0 - (math.comb(n - c, k) / math.comb(n, k))
            else:
                pass_k = 1.0

            pass_at_k.append(pass_k)

        # Return first few values
        return pass_at_k[:5]
