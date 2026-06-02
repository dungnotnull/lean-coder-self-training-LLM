"""Tests for model registry."""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime

from src.registry.manager import ModelRegistry, CheckpointInfo, EvalResult


@pytest.fixture
def temp_registry(tmp_path):
    """Create temporary registry."""
    registry_path = tmp_path / "test_registry.json"
    return ModelRegistry(str(registry_path))


def test_registry_init(temp_registry):
    """Test registry initialization."""
    assert temp_registry.registry_path.exists()
    assert len(temp_registry.checkpoints) == 0
    assert temp_registry.best_pointer is None


def test_register_checkpoint(temp_registry):
    """Test checkpoint registration."""
    info = CheckpointInfo(
        id="test-001",
        source="base",
        base_model="test/model",
        dataset_version=None,
        path="/test/path",
        eval=EvalResult(
            pass_at_1=0.5,
            pass_at_k=[0.5, 0.7],
            latency_tok_s=10.0,
            size_mb=1000,
            memory_mb=2000,
        ),
        is_best=True,
        created_at=datetime.now(),
    )

    temp_registry.register_checkpoint(info)

    assert len(temp_registry.checkpoints) == 1
    assert temp_registry.get_checkpoint("test-001") is not None


def test_promote_if_better(temp_registry):
    """Test promotion logic."""
    # Register first checkpoint
    info1 = CheckpointInfo(
        id="test-001",
        source="base",
        base_model="test/model",
        dataset_version=None,
        path="/test/path1",
        eval=EvalResult(
            pass_at_1=0.5,
            pass_at_k=[0.5, 0.7],
            latency_tok_s=10.0,
            size_mb=1000,
            memory_mb=2000,
        ),
        is_best=True,
        created_at=datetime.now(),
    )
    temp_registry.register_checkpoint(info1)

    # Register better checkpoint
    info2 = CheckpointInfo(
        id="test-002",
        source="qlora-sft",
        base_model="test/model",
        dataset_version="ds-v1",
        path="/test/path2",
        eval=EvalResult(
            pass_at_1=0.6,
            pass_at_k=[0.6, 0.8],
            latency_tok_s=10.0,
            size_mb=1000,
            memory_mb=2000,
        ),
        is_best=False,
        created_at=datetime.now(),
    )
    temp_registry.register_checkpoint(info2)

    # Promote better checkpoint
    promoted = temp_registry.promote_if_better("test-002")

    assert promoted is True
    assert temp_registry.best_pointer == "test-002"
    assert temp_registry.get_checkpoint("test-001").is_best is False
    assert temp_registry.get_checkpoint("test-002").is_best is True
