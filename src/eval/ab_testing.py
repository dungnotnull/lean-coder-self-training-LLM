"""A/B testing framework for comparing models side-by-side."""

from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import json


class TestStatus(Enum):
    """Status of A/B test."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ModelVariant:
    """A model variant for testing."""

    variant_id: str
    checkpoint_id: str
    model_path: str
    config: Dict = None


@dataclass
class TestCase:
    """A test case for A/B testing."""

    case_id: str
    prompt: str
    expected_output: Optional[str] = None
    category: str = "general"


@dataclass
class TestResult:
    """Result from testing a model on a case."""

    variant_id: str
    case_id: str
    response: str
    passed: bool
    latency_ms: float
    error: Optional[str] = None


@dataclass
class ABTest:
    """An A/B test comparing models."""

    test_id: str
    name: str
    variants: List[ModelVariant]
    test_cases: List[TestCase]
    status: TestStatus = TestStatus.PENDING
    results: List[TestResult] = None
    winner: Optional[str] = None


class ABTestFramework:
    """Framework for running A/B tests."""

    def __init__(self, storage_path: str = None):
        """Initialize A/B test framework.

        Args:
            storage_path: Path for storing test results
        """
        self.storage_path = storage_path
        self.tests: Dict[str, ABTest] = {}
        self._load_tests()

    def create_test(
        self,
        test_id: str,
        name: str,
        variant_a_path: str,
        variant_b_path: str,
        test_cases: List[TestCase],
    ) -> ABTest:
        """Create a new A/B test.

        Args:
            test_id: Test identifier
            name: Test name
            variant_a_path: Path to model A
            variant_b_path: Path to model B
            test_cases: Test cases to run

        Returns:
            Created ABTest
        """
        variants = [
            ModelVariant("A", test_id + "_A", variant_a_path),
            ModelVariant("B", test_id + "_B", variant_b_path),
        ]

        test = ABTest(
            test_id=test_id,
            name=name,
            variants=variants,
            test_cases=test_cases,
        )

        self.tests[test_id] = test
        self._save_test(test)

        return test

    def run_test(self, test_id: str) -> Dict:
        """Run an A/B test.

        Args:
            test_id: Test identifier

        Returns:
            Test results summary
        """
        test = self.tests.get(test_id)
        if not test:
            raise ValueError(f"Test not found: {test_id}")

        test.status = TestStatus.RUNNING
        test.results = []

        print(f"\n{'='*60}")
        print(f"Running A/B Test: {test.name}")
        print(f"{'='*60}")

        # Run each variant on each test case
        for variant in test.variants:
            print(f"\nTesting Variant {variant.variant_id}...")

            for case in test.test_cases:
                result = self._test_variant(variant, case)
                test.results.append(result)

        # Determine winner
        test.winner = self._determine_winner(test)
        test.status = TestStatus.COMPLETED

        self._save_test(test)

        summary = {
            "test_id": test_id,
            "name": test.name,
            "status": test.status.value,
            "winner": test.winner,
            "summary": self._generate_summary(test),
        }

        print(f"\n{'='*60}")
        print(f"Test Complete. Winner: Variant {test.winner}")
        print(f"{'='*60}\n")

        return summary

    def _test_variant(self, variant: ModelVariant, case: TestCase) -> TestResult:
        """Test a variant on a case.

        Args:
            variant: Model variant
            case: Test case

        Returns:
            Test result
        """
        import time

        start_time = time.time()

        try:
            # Generate response (placeholder)
            response = self._generate_response(variant, case.prompt)

            latency = (time.time() - start_time) * 1000

            # Evaluate response
            passed = self._evaluate_response(response, case)

            return TestResult(
                variant_id=variant.variant_id,
                case_id=case.case_id,
                response=response,
                passed=passed,
                latency_ms=latency,
                error=None,
            )

        except Exception as e:
            latency = (time.time() - start_time) * 1000

            return TestResult(
                variant_id=variant.variant_id,
                case_id=case.case_id,
                response="",
                passed=False,
                latency_ms=latency,
                error=str(e),
            )

    def _generate_response(self, variant: ModelVariant, prompt: str) -> str:
        """Generate response from model.

        Args:
            variant: Model variant
            prompt: Input prompt

        Returns:
            Generated response
        """
        # Placeholder: would load actual model and generate
        return f"Response from {variant.variant_id} for: {prompt[:50]}..."

    def _evaluate_response(self, response: str, case: TestCase) -> bool:
        """Evaluate if response is correct.

        Args:
            response: Generated response
            case: Test case

        Returns:
            True if response passes
        """
        # Placeholder: would run actual evaluation
        return len(response) > 20 and "error" not in response.lower()

    def _determine_winner(self, test: ABTest) -> str:
        """Determine winning variant.

        Args:
            test: Completed test

        Returns:
            Winning variant ID
        """
        scores = {}

        for variant in test.variants:
            variant_results = [r for r in test.results if r.variant_id == variant.variant_id]

            if not variant_results:
                scores[variant.variant_id] = 0
                continue

            pass_rate = sum(1 for r in variant_results if r.passed) / len(variant_results)
            avg_latency = sum(r.latency_ms for r in variant_results) / len(variant_results)

            # Score: pass rate - latency penalty
            latency_penalty = avg_latency / 10000  # 10s latency = 1 point penalty
            scores[variant.variant_id] = pass_rate - latency_penalty

        return max(scores, key=scores.get)

    def _generate_summary(self, test: ABTest) -> Dict:
        """Generate test summary.

        Args:
            test: Completed test

        Returns:
            Summary dict
        """
        summary = {"variants": {}, "cases": len(test.test_cases)}

        for variant in test.variants:
            variant_results = [r for r in test.results if r.variant_id == variant.variant_id]

            if variant_results:
                passed = sum(1 for r in variant_results if r.passed)
                avg_latency = sum(r.latency_ms for r in variant_results) / len(variant_results)

                summary["variants"][variant.variant_id] = {
                    "passed": passed,
                    "total": len(variant_results),
                    "pass_rate": passed / len(variant_results),
                    "avg_latency_ms": avg_latency,
                }

        return summary

    def compare_results(self, test_id: str) -> Dict:
        """Get detailed comparison of test results.

        Args:
            test_id: Test identifier

        Returns:
            Comparison dict
        """
        test = self.tests.get(test_id)
        if not test:
            raise ValueError(f"Test not found: {test_id}")

        comparison = {"test_id": test_id, "name": test.name, "case_comparison": []}

        for case in test.test_cases:
            case_results = [r for r in test.results if r.case_id == case.case_id]

            case_comp = {
                "case_id": case.case_id,
                "category": case.category,
                "variants": {},
            }

            for result in case_results:
                case_comp["variants"][result.variant_id] = {
                    "passed": result.passed,
                    "latency_ms": result.latency_ms,
                    "response_preview": result.response[:100],
                }

            comparison["case_comparison"].append(case_comp)

        return comparison

    def _save_test(self, test: ABTest):
        """Save test to storage.

        Args:
            test: Test to save
        """
        if not self.storage_path:
            return

        from pathlib import Path

        Path(self.storage_path).parent.mkdir(parents=True, exist_ok=True)

        with open(self.storage_path, "w") as f:
            json.dump(
                {
                    "test_id": test.test_id,
                    "name": test.name,
                    "variants": [
                        {
                            "variant_id": v.variant_id,
                            "checkpoint_id": v.checkpoint_id,
                            "model_path": v.model_path,
                        }
                        for v in test.variants
                    ],
                    "test_cases": [
                        {
                            "case_id": c.case_id,
                            "prompt": c.prompt,
                            "category": c.category,
                        }
                        for c in test.test_cases
                    ],
                    "status": test.status.value,
                    "results": [
                        {
                            "variant_id": r.variant_id,
                            "case_id": r.case_id,
                            "response": r.response,
                            "passed": r.passed,
                            "latency_ms": r.latency_ms,
                            "error": r.error,
                        }
                        for r in test.results or []
                    ],
                    "winner": test.winner,
                },
                f,
                indent=2,
            )

    def _load_tests(self):
        """Load tests from storage."""
        if not self.storage_path:
            return

        from pathlib import Path

        if not Path(self.storage_path).exists():
            return

        with open(self.storage_path) as f:
            data = json.load(f)

        # Reconstruct ABTest (simplified)
        pass

    def list_tests(self) -> List[Dict]:
        """List all tests.

        Returns:
            List of test summaries
        """
        return [
            {
                "test_id": test.test_id,
                "name": test.name,
                "status": test.status.value,
                "variants": len(test.variants),
                "cases": len(test.test_cases),
            }
            for test in self.tests.values()
        ]
