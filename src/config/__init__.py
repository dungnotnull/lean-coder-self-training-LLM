"""Configuration management for LeanCoder."""

from pathlib import Path
from typing import Any
import os
from dotenv import load_dotenv
import yaml


class Config:
    """Central configuration manager."""

    def __init__(self):
        """Load environment variables and configs."""
        load_dotenv()

        self.root_dir = Path(__file__).parent.parent.parent
        self.artifacts_dir = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))
        self.outputs_dir = Path(os.getenv("OUTPUTS_DIR", "outputs"))
        self.registry_path = Path(os.getenv("REGISTRY_PATH", "registry/models.json"))
        self.data_dir = Path(os.getenv("DATA_DIR", "data"))

        # Create directories
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)

    def load_yaml(self, path: str) -> dict[str, Any]:
        """Load YAML config file."""
        full_path = self.root_dir / path
        with open(full_path) as f:
            return yaml.safe_load(f)

    @property
    def hf_token(self) -> str | None:
        """Get HuggingFace token."""
        return os.getenv("HF_TOKEN")

    @property
    def cuda_devices(self) -> str:
        """Get CUDA device configuration."""
        return os.getenv("CUDA_VISIBLE_DEVICES", "0")

    @property
    def max_gpu_mem_gb(self) -> int:
        """Get maximum GPU memory in GB."""
        return int(os.getenv("MAX_GPU_MEM_GB", "80"))


# Global config instance
config = Config()
