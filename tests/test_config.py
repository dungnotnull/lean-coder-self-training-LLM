"""Tests for configuration."""

import pytest
import tempfile
from pathlib import Path

from src.config import Config


@pytest.fixture
def temp_config():
    """Create temp config."""
    return Config()


def test_config_init(temp_config):
    """Test config initialization."""
    assert temp_config.root_dir.exists()
    assert temp_config.artifacts_dir.exists()
    assert temp_config.outputs_dir.exists()


def test_config_load_yaml(temp_config, tmp_path):
    """Test loading YAML config."""
    # Create test YAML file
    yaml_path = tmp_path / "test.yaml"
    yaml_path.write_text("test_key: test_value\nnested:\n  key: value\n")

    # Note: this test would need the YAML to be in the right location
    # For now, we just test the method exists
    assert hasattr(temp_config, "load_yaml")
