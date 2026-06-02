"""Model abstraction layer for multiple model architectures."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path
from enum import Enum


class ModelType(Enum):
    """Supported model types."""

    QWEN = "qwen"
    LLAMA = "llama"
    CODEGEMMA = "codegemma"
    STARCODER = "starcoder"
    DEEPSEEK = "deepseek"


@dataclass
class ModelConfig:
    """Generic model configuration."""

    model_type: ModelType
    repo_id: str
    revision: str = "main"
    context_length: int = 2048
    trust_remote_code: bool = True


@dataclass
class GenerationConfig:
    """Text generation configuration."""

    max_new_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.95
    top_k: int = 50
    do_sample: bool = True
    num_return_sequences: int = 1
    repetition_penalty: float = 1.0


class BaseModel(ABC):
    """Base class for all model implementations."""

    def __init__(self, config: ModelConfig):
        """Initialize model.

        Args:
            config: Model configuration
        """
        self.config = config
        self.model = None
        self.tokenizer = None

    @abstractmethod
    def load(self, model_path: str):
        """Load model from path.

        Args:
            model_path: Path to model weights
        """
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        generation_config: GenerationConfig = None,
    ) -> str:
        """Generate text from prompt.

        Args:
            prompt: Input prompt
            generation_config: Generation configuration

        Returns:
            Generated text
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict:
        """Get model information.

        Returns:
            Model info dict
        """
        pass

    def format_prompt(self, prompt: str, chat_format: bool = True) -> str:
        """Format prompt for model.

        Args:
            prompt: Raw prompt
            chat_format: Use chat format

        Returns:
            Formatted prompt
        """
        if chat_format:
            return self._chat_format(prompt)
        return prompt

    def _chat_format(self, prompt: str) -> str:
        """Format prompt as chat.

        Args:
            prompt: Raw prompt

        Returns:
            Chat-formatted prompt
        """
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}({self.config.model_type.value})"


class QwenModel(BaseModel):
    """Qwen/Coder model implementation."""

    def load(self, model_path: str):
        """Load Qwen model.

        Args:
            model_path: Path to model weights
        """
        # Placeholder: would load actual Qwen model
        print(f"Loading Qwen model from {model_path}")
        self.model = f"QwenModel({model_path})"
        self.tokenizer = f"QwenTokenizer"

    def generate(
        self,
        prompt: str,
        generation_config: GenerationConfig = None,
    ) -> str:
        """Generate text.

        Args:
            prompt: Input prompt
            generation_config: Generation config

        Returns:
            Generated text
        """
        config = generation_config or GenerationConfig()

        formatted_prompt = self.format_prompt(prompt)
        # Placeholder: would generate actual text
        return f"[Qwen Response to: {formatted_prompt[:50]}...]"

    def get_model_info(self) -> Dict:
        """Get model info.

        Returns:
            Model info
        """
        return {
            "type": "qwen",
            "context_length": self.config.context_length,
            "architecture": "Qwen2ForCausalLM",
        }


class LlamaModel(BaseModel):
    """Llama model implementation."""

    def load(self, model_path: str):
        """Load Llama model.

        Args:
            model_path: Path to model weights
        """
        print(f"Loading Llama model from {model_path}")
        self.model = f"LlamaModel({model_path})"
        self.tokenizer = f"LlamaTokenizer"

    def generate(
        self,
        prompt: str,
        generation_config: GenerationConfig = None,
    ) -> str:
        """Generate text.

        Args:
            prompt: Input prompt
            generation_config: Generation config

        Returns:
            Generated text
        """
        config = generation_config or GenerationConfig()
        formatted_prompt = self.format_prompt(prompt)
        return f"[Llama Response to: {formatted_prompt[:50]}...]"

    def get_model_info(self) -> Dict:
        """Get model info.

        Returns:
            Model info
        """
        return {
            "type": "llama",
            "context_length": self.config.context_length,
            "architecture": "LlamaForCausalLM",
        }


class ModelFactory:
    """Factory for creating model instances."""

    _model_classes = {
        ModelType.QWEN: QwenModel,
        ModelType.LLAMA: LlamaModel,
        ModelType.CODEGEMMA: QwenModel,  # Reuse Qwen for now
        ModelType.STARCODER: QwenModel,
        ModelType.DEEPSEEK: QwenModel,
    }

    @classmethod
    def create_model(
        cls,
        config: ModelConfig,
    ) -> BaseModel:
        """Create model instance.

        Args:
            config: Model configuration

        Returns:
            Model instance
        """
        model_class = cls._model_classes.get(config.model_type)

        if model_class is None:
            raise ValueError(f"Unsupported model type: {config.model_type}")

        return model_class(config)

    @classmethod
    def register_model(cls, model_type: ModelType, model_class: type):
        """Register a new model class.

        Args:
            model_type: Model type enum
            model_class: Model class
        """
        cls._model_classes[model_type] = model_class

    @classmethod
    def from_repo_id(cls, repo_id: str) -> BaseModel:
        """Create model from HuggingFace repo ID.

        Args:
            repo_id: HuggingFace repo ID

        Returns:
            Model instance
        """
        # Detect model type from repo_id
        if "Qwen" in repo_id or "qwen" in repo_id:
            model_type = ModelType.QWEN
        elif "Llama" in repo_id or "llama" in repo_id:
            model_type = ModelType.LLAMA
        else:
            model_type = ModelType.QWEN  # Default

        config = ModelConfig(
            model_type=model_type,
            repo_id=repo_id,
        )

        return cls.create_model(config)


def load_model(
    model_path: str,
    model_type: str = None,
) -> BaseModel:
    """Convenience function to load a model.

    Args:
        model_path: Path to model
        model_type: Optional model type string

    Returns:
        Loaded model
    """
    # Detect model type from path if not specified
    if model_type is None:
        if "Qwen" in model_path or "qwen" in model_path:
            model_type = "qwen"
        elif "Llama" in model_path or "llama" in model_path:
            model_type = "llama"
        else:
            model_type = "qwen"

    config = ModelConfig(
        model_type=ModelType(model_type),
        repo_id=model_path,
    )

    model = ModelFactory.create_model(config)
    model.load(model_path)

    return model
