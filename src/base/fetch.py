"""Base model download and license verification."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import os

from huggingface_hub import login, snapshot_download
from transformers import AutoConfig, AutoTokenizer
from transformers.utils import is_accelerate_available


class LicenseError(Exception):
    """Raised when model license doesn't match expected license."""

    pass


class DownloadError(Exception):
    """Raised when model download fails."""

    pass


@dataclass
class ModelInfo:
    """Recorded information about a downloaded model."""

    repo_id: str
    revision: str
    license: str
    architecture: str
    context_length: int
    vocab_size: int
    hidden_size: int
    num_layers: int
    num_attention_heads: int
    path: Path
    size_mb: float

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "repo_id": self.repo_id,
            "revision": self.revision,
            "license": self.license,
            "architecture": self.architecture,
            "context_length": self.context_length,
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_attention_heads": self.num_attention_heads,
            "path": str(self.path),
            "size_mb": self.size_mb,
        }


def download_base_model(config_dict: dict, hf_token: Optional[str] = None) -> ModelInfo:
    """Download base model from HuggingFace with license verification.

    Args:
        config_dict: Configuration dict from config/base.yaml
        hf_token: Optional HuggingFace token (overrides env var)

    Returns:
        ModelInfo with recorded metadata

    Raises:
        LicenseError: If model license doesn't match expected
        DownloadError: If download fails
    """
    model_config = config_dict["model"]
    download_config = config_dict["download"]

    repo_id = model_config["repo_id"]
    revision = model_config.get("revision", "main")
    expected_license = model_config["expected_license"]
    trust_remote_code = model_config.get("trust_remote_code", False)
    save_path = download_config["save_path"]
    use_auth_token = download_config.get("use_auth_token", False)

    # Login if token provided
    if use_auth_token and hf_token:
        login(token=hf_token)

    # Get model config first (lightweight)
    try:
        cfg = AutoConfig.from_pretrained(
            repo_id,
            revision=revision,
            token=hf_token if use_auth_token else None,
            trust_remote_code=trust_remote_code,
        )
    except Exception as e:
        raise DownloadError(f"Failed to load model config: {e}")

    # Check license
    actual_license = getattr(cfg, "license", None)
    if actual_license is None:
        # Try to get from repo metadata
        from huggingface_hub import model_info

        try:
            info = model_info(repo_id, token=hf_token if use_auth_token else None)
            actual_license = info.cardData.get("license", "unknown")
        except Exception:
            actual_license = "unknown"

    if expected_license and actual_license != expected_license:
        raise LicenseError(
            f"License mismatch: expected '{expected_license}', got '{actual_license}'"
        )

    # Download model weights
    try:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        model_path = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=save_path,
            local_dir_use_symlinks=False,
            token=hf_token if use_auth_token else None,
            trust_remote_code=trust_remote_code,
        )
    except Exception as e:
        raise DownloadError(f"Failed to download model: {e}")

    # Load tokenizer to get vocab size
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            token=hf_token if use_auth_token else None,
            trust_remote_code=trust_remote_code,
        )
        vocab_size = tokenizer.vocab_size
    except Exception as e:
        vocab_size = getattr(cfg, "vocab_size", 0)

    # Calculate size
    size_mb = _calculate_directory_size(Path(model_path))

    # Extract architecture details
    architecture = cfg.architectures[0] if cfg.architectures else "Unknown"

    return ModelInfo(
        repo_id=repo_id,
        revision=revision,
        license=actual_license,
        architecture=architecture,
        context_length=getattr(cfg, "max_position_embeddings", 0),
        vocab_size=vocab_size,
        hidden_size=getattr(cfg, "hidden_size", 0),
        num_layers=getattr(cfg, "num_hidden_layers", 0),
        num_attention_heads=getattr(cfg, "num_attention_heads", 0),
        path=Path(model_path),
        size_mb=size_mb,
    )


def _calculate_directory_size(path: Path) -> float:
    """Calculate directory size in MB."""
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total / (1024 * 1024)
