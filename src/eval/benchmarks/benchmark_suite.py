"""Unified benchmark suite for comprehensive evaluation."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime
import time

from .human_eval import HumanEvalBenchmark, HumanEvalProblem
from .mbpp import MBPPBenchmark, MBPPProblem
from .leetcode import LeetCodeBenchmark, LeetCodeProblem


@dataclass
class BenchmarkResult:
    """Results from running a benchmark."""

    suite_name: str
    total_problems: int
    attempted: int
    passed: int
    failed: int
    pass_at_1: float
    pass_at_10: float
    latency_avg: float
    total_time: float
    details: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "suite_name": self.suite_name,
            "total_problems": self.total_problems,
            "attempted": self.attempted,
            "passed": self.passed,
            "failed": self.failed,
            "pass_at_1": self.pass_at_1,
            "pass_at_10": self.pass_at_10,
            "latency_avg": self.latency_avg,
            "total_time": self.total_time,
            "details": self.details,
        }


class BenchmarkSuite:
    """Unified benchmark suite combining multiple benchmarks."""

    def __init__(self):
        """Initialize benchmark suite."""
        self.human_eval = HumanEvalBenchmark()
        self.mbpp = MBPPBenchmark()
        self.leetcode = LeetCodeBenchmark()

        self.benchmarks = {
            "humaneval": self.human_eval,
            "mbpp": self.mbpp,
            "leetcode": self.leetcode,
        }

    def run_benchmark(
        self,
        suite_name: str,
        model,
        n_samples: int = 10,
        timeout: int = 10,
    ) -> BenchmarkResult:
        """Run a specific benchmark.

        Args:
            suite_name: Name of benchmark suite
            model: Model to evaluate
            n_samples: Number of samples per problem
            timeout: Timeout per problem in seconds

        Returns:
            BenchmarkResult
        """
        if suite_name not in self.benchmarks:
            raise ValueError(f"Unknown benchmark: {suite_name}")

        benchmark = self.benchmarks[suite_name]
        problems = benchmark.get_problems()

        start_time = time.time()
        results = []

        for problem in problems:
            # Generate solution
            problem_start = time.time()

            # Extract prompt based on problem type
            if isinstance(problem, HumanEvalProblem):
                prompt = problem.prompt
                entry_point = problem.entry_point
                tests = problem.test
            elif isinstance(problem, MBPPProblem):
                prompt = problem.text
                entry_point = None
                tests = problem.test_list
            elif isinstance(problem, LeetCodeProblem):
                prompt = problem.description + "\n" + problem.starter_code
                entry_point = None
                tests = []
            else:
                continue

            # Generate (placeholder - would use actual model)
            # solution = model.generate(prompt, max_tokens=512)
            solution = "pass"  # Placeholder

            problem_time = time.time() - problem_start

            # Test solution (would use sandbox)
            passed = self._test_solution(solution, tests)

            results.append({
                "problem_id": getattr(problem, "task_id", getattr(problem, "problem_id", "unknown")),
                "passed": passed,
                "time": problem_time,
            })

        total_time = time.time() - start_time

        # Calculate metrics
        passed_count = sum(1 for r in results if r["passed"])
        total_count = len(results)

        # Calculate pass@k (simplified)
        pass_at_1 = passed_count / total_count if total_count > 0 else 0.0
        pass_at_10 = min(1.0, pass_at_1 * 1.5)  # Placeholder formula

        avg_latency = sum(r["time"] for r in results) / total_count if total_count > 0 else 0.0

        return BenchmarkResult(
            suite_name=suite_name,
            total_problems=len(problems),
            attempted=total_count,
            passed=passed_count,
            failed=total_count - passed_count,
            pass_at_1=pass_at_1,
            pass_at_10=pass_at_10,
            latency_avg=avg_latency,
            total_time=total_time,
            details=results,
        )

    def _test_solution(self, solution: str, tests: List[str]) -> bool:
        """Test a solution against test cases.

        Args:
            solution: Generated solution code
            tests: List of test assertions

        Returns:
            True if all tests pass
        """
        # Placeholder: would use sandbox to execute
        # For now, return True if solution exists
        return bool(solution and solution != "pass")

    def run_all_benchmarks(
        self,
        model,
        n_samples: int = 10,
    ) -> Dict[str, BenchmarkResult]:
        """Run all benchmarks.

        Args:
            model: Model to evaluate
            n_samples: Number of samples per problem

        Returns:
            Dict mapping suite names to results
        """
        results = {}

        for suite_name in self.benchmarks.keys():
            print(f"\n{'='*60}")
            print(f"Running {suite_name.upper()} benchmark")
            print(f"{'='*60}")

            try:
                result = self.run_benchmark(suite_name, model, n_samples)
                results[suite_name] = result

                print(f"Results: {result.passed}/{result.total_problems} passed")
                print(f"Pass@1: {result.pass_at_1:.2%}")
                print(f"Time: {result.total_time:.2f}s")

            except Exception as e:
                print(f"Error running {suite_name}: {e}")

        return results

    def get_summary(self, results: Dict[str, BenchmarkResult]) -> Dict:
        """Get summary of all benchmark results.

        Args:
            results: Dict of benchmark results

        Returns:
            Summary statistics
        """
        summary = {
            "total_suites": len(results),
            "total_problems": sum(r.total_problems for r in results.values()),
            "total_passed": sum(r.passed for r in results.values()),
            "overall_pass_rate": 0.0,
            "suites": {},
        }

        if summary["total_problems"] > 0:
            summary["overall_pass_rate"] = summary["total_passed"] / summary["total_problems"]

        for suite_name, result in results.items():
            summary["suites"][suite_name] = {
                "pass_at_1": result.pass_at_1,
                "passed": result.passed,
                "total": result.total_problems,
            }

        return summary

    def list_problems(self, suite_name: str = None) -> List:
        """List problems in a suite or all suites.

        Args:
            suite_name: Optional suite name

        Returns:
            List of problems
        """
        if suite_name:
            return self.benchmarks[suite_name].get_problems()
        else:
            all_problems = []
            for benchmark in self.benchmarks.values():
                all_problems.extend(benchmark.get_problems())
            return all_problems

    def __repr__(self) -> str:
        """String representation."""
        total = sum(len(b) for b in self.benchmarks.values())
        return f"BenchmarkSuite({total} problems across {len(self.benchmarks)} suites)"
