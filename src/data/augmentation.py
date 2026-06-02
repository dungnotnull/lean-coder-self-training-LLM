"""Data augmentation pipeline for synthesizing new training examples."""

from typing import List, Dict, Optional
from dataclasses import dataclass
import random


@dataclass
class AugmentedExample:
    """Augmented training example."""

    prompt: str
    response: str
    source: str
    augmentation_type: str


class DataAugmenter:
    """Augment training examples with variations."""

    def __init__(self, config: Dict = None):
        """Initialize data augmenter.

        Args:
            config: Augmentation configuration
        """
        self.config = config or {}

    def paraphrase(self, example: Dict) -> List[AugmentedExample]:
        """Generate paraphrased variations.

        Args:
            example: Original example with prompt and response

        Returns:
            List of augmented examples
        """
        variations = []

        # Paraphrase prompt
        paraphrases = self._paraphrase_text(example["prompt"])

        for para_prompt in paraphrases:
            variations.append(
                AugmentedExample(
                    prompt=para_prompt,
                    response=example["response"],
                    source=example.get("source", "unknown"),
                    augmentation_type="paraphrase",
                )
            )

        return variations

    def synthesize_variations(self, example: Dict, n_variations: int = 3) -> List[AugmentedExample]:
        """Synthesize solution variations.

        Args:
            example: Original example
            n_variations: Number of variations to generate

        Returns:
            List of augmented examples
        """
        variations = []

        # Generate alternative solutions
        if "code" in example:
            solutions = self._generate_code_variations(example["code"], n_variations)
            for solution in solutions:
                variations.append(
                    AugmentedExample(
                        prompt=example["prompt"],
                        response=solution,
                        source=example.get("source", "unknown"),
                        augmentation_type="solution_variation",
                    )
                )

        return variations

    def generate_edge_cases(self, example: Dict) -> List[AugmentedExample]:
        """Generate edge case examples.

        Args:
            example: Original example

        Returns:
            List of edge case examples
        """
        edge_cases = []

        # Generate edge cases based on problem type
        if "function" in example.get("tags", []):
            # For function problems, add edge cases
            edge_cases.extend(self._generate_function_edge_cases(example))
        elif "algorithm" in example.get("tags", []):
            edge_cases.extend(self._generate_algorithm_edge_cases(example))

        return edge_cases

    def _paraphrase_text(self, text: str, n_variants: int = 3) -> List[str]:
        """Generate paraphrased versions of text.

        Args:
            text: Original text
            n_variants: Number of variants

        Returns:
            List of paraphrased texts
        """
        # Placeholder: In real implementation, would use LLM or rule-based paraphrasing
        variants = [text]

        # Simple rule-based paraphrasing
        paraphrases = [
            text.replace("write", "create"),
            text.replace("function", "method"),
            text.replace("implement", "develop"),
        ]

        for para in paraphrases:
            if para != text and para not in variants:
                variants.append(para)
                if len(variants) >= n_variants:
                    break

        return variants[:n_variants]

    def _generate_code_variations(self, code: str, n_variations: int) -> List[str]:
        """Generate code variations.

        Args:
            code: Original code
            n_variations: Number of variations

        Returns:
            List of code variations
        """
        # Placeholder: Would use code transformation rules
        variations = [code]

        # Simple variations
        if "for" in code:
            # Try list comprehension variation
            variations.append(code.replace("for ", "# list comprehension"))

        return variations[:n_variations]

    def _generate_function_edge_cases(self, example: Dict) -> List[AugmentedExample]:
        """Generate edge cases for function problems."""
        edge_cases = []

        # Common edge cases
        edge_case_prompts = [
            example["prompt"] + " Handle empty input.",
            example["prompt"] + " Handle null/None values.",
            example["prompt"] + " Optimize for large inputs.",
        ]

        for prompt in edge_case_prompts:
            edge_cases.append(
                AugmentedExample(
                    prompt=prompt,
                    response=example.get("response", ""),
                    source=example.get("source", "unknown"),
                    augmentation_type="edge_case",
                )
            )

        return edge_cases

    def _generate_algorithm_edge_cases(self, example: Dict) -> List[AugmentedExample]:
        """Generate edge cases for algorithm problems."""
        edge_cases = []

        # Algorithm-specific edge cases
        edge_case_prompts = [
            example["prompt"] + " What's the time complexity?",
            example["prompt"] + " Can you optimize space complexity?",
        ]

        for prompt in edge_case_prompts:
            edge_cases.append(
                AugmentedExample(
                    prompt=prompt,
                    response=example.get("response", ""),
                    source=example.get("source", "unknown"),
                    augmentation_type="edge_case",
                )
            )

        return edge_cases

    def augment_dataset(
        self,
        examples: List[Dict],
        target_size: int,
    ) -> List[AugmentedExample]:
        """Augment dataset to target size.

        Args:
            examples: Original examples
            target_size: Target number of examples

        Returns:
            List of original + augmented examples
        """
        augmented = []

        # Add originals
        for ex in examples:
            augmented.append(
                AugmentedExample(
                    prompt=ex["prompt"],
                    response=ex["response"],
                    source=ex.get("source", "unknown"),
                    augmentation_type="original",
                )
            )

        # Augment until target size reached
        while len(augmented) < target_size:
            # Random example to augment
            example = random.choice(examples)

            # Random augmentation type
            aug_type = random.choice(["paraphrase", "variation", "edge_case"])

            if aug_type == "paraphrase":
                augmented.extend(self.paraphrase(example))
            elif aug_type == "variation":
                augmented.extend(self.synthesize_variations(example))
            elif aug_type == "edge_case":
                augmented.extend(self.generate_edge_cases(example))

        return augmented[:target_size]


class SyntheticDataGenerator:
    """Generate synthetic coding problems."""

    def __init__(self, templates_path: str = None):
        """Initialize synthetic data generator.

        Args:
            templates_path: Path to problem templates
        """
        self.templates_path = templates_path
        self.templates = []
        self._load_templates()

    def _load_templates(self):
        """Load problem templates."""
        # Default templates
        self.templates = [
            {
                "type": "function",
                "template": "Write a {function_type} function to {action} {input_type}.",
                "params": {
                    "function_type": ["helper", "utility", "validator"],
                    "action": ["check", "validate", "transform", "convert"],
                    "input_type": ["strings", "numbers", "lists", "dictionaries"],
                },
            },
            {
                "type": "algorithm",
                "template": "Implement {algorithm} for {data_structure}.",
                "params": {
                    "algorithm": ["sorting", "searching", "traversal", "optimization"],
                    "data_structure": ["arrays", "linked lists", "trees", "graphs"],
                },
            },
        ]

    def generate_problem(self, template_id: int = None) -> Dict:
        """Generate a synthetic problem.

        Args:
            template_id: Specific template to use

        Returns:
            Generated problem
        """
        template = random.choice(self.templates) if template_id is None else self.templates[template_id]

        # Fill template
        prompt = template["template"]
        for param, values in template["params"].items():
            value = random.choice(values)
            prompt = prompt.replace(f"{{{param}}}", value)

        # Generate corresponding solution (placeholder)
        solution = self._generate_solution(prompt, template["type"])

        return {
            "prompt": prompt,
            "response": solution,
            "source": "synthetic",
            "type": template["type"],
        }

    def _generate_solution(self, prompt: str, problem_type: str) -> str:
        """Generate solution for problem.

        Args:
            prompt: Problem prompt
            problem_type: Type of problem

        Returns:
            Solution code
        """
        # Placeholder: would use LLM to generate solution
        return f"# Solution for: {prompt}\ndef solve():\n    pass"

    def generate_batch(self, n_problems: int) -> List[Dict]:
        """Generate batch of problems.

        Args:
            n_problems: Number of problems to generate

        Returns:
            List of generated problems
        """
        return [self.generate_problem() for _ in range(n_problems)]
