"""Tests for evaluation harness."""

import pytest

from src.eval.harness import EvalHarness, Problem, EvalResult


@pytest.fixture
def harness_config():
    """Create test harness config."""
    return {
        "sandbox": {"timeout_seconds": 10, "memory_limit_mb": 256},
        "metrics": {"n_samples": [5], "temperature": 0.2},
    }


@pytest.fixture
def sample_problems():
    """Create sample problems."""
    return [
        Problem(
            id="test_001",
            description="Add two numbers",
            starter_code="def add(a, b):\n    return a + b\n",
            tests=["assert add(1, 2) == 3", "assert add(-1, 1) == 0"],
        ),
        Problem(
            id="test_002",
            description="Subtract two numbers",
            starter_code="def sub(a, b):\n    return a - b\n",
            tests=["assert sub(5, 3) == 2"],
        ),
    ]


def test_harness_init(harness_config):
    """Test harness initialization."""
    harness = EvalHarness(harness_config)
    assert harness.timeout == 10
    assert harness.memory_limit == 256


def test_harness_load_problems_empty(harness_config):
    """Test loading problems when file doesn't exist."""
    harness = EvalHarness(harness_config)
    problems = harness.load_problems("/nonexistent/path.json")

    # Should return sample problems
    assert len(problems) > 0
    assert all(isinstance(p, Problem) for p in problems)


def test_harness_run_eval(harness_config, sample_problems):
    """Test running evaluation."""
    harness = EvalHarness(harness_config)

    # Mock model that returns the starter code
    class MockModel:
        def generate(self, prompt, **kwargs):
            return "pass"

    mock_model = MockModel()

    result = harness.run_eval(mock_model, sample_problems, n_samples=2)

    assert isinstance(result, EvalResult)
    assert result.suite == "eval"
    assert result.problems_total == len(sample_problems)
    assert len(result.details) == len(sample_problems)
