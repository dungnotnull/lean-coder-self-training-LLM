"""Data formatting for chat and code templates."""

from typing import Any


class ChatFormatter:
    """Formats data into chat templates for training."""

    def __init__(self, config: dict):
        """Initialize formatter.

        Args:
            config: Formatting configuration
        """
        self.config = config
        self.template = config.get("template", "default")

    def format_chat(self, prompt: str, response: str) -> str:
        """Format a prompt-response pair as chat.

        Args:
            prompt: User prompt
            response: Model response

        Returns:
            Formatted chat string
        """
        if self.template == "qwen":
            return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
        else:
            # Default format
            return f"User: {prompt}\nAssistant: {response}"

    def format_code(self, problem: str, solution: str) -> str:
        """Format a coding problem with solution.

        Args:
            problem: Problem description
            solution: Code solution

        Returns:
            Formatted string
        """
        return f"Problem: {problem}\n\nSolution:\n```python\n{solution}\n```"
