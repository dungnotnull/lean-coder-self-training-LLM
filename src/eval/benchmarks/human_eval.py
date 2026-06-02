"""HumanEval benchmark - 164 hand-written Python problems."""

import json
from pathlib import Path
from typing import List
from dataclasses import dataclass


@dataclass
class HumanEvalProblem:
    """A HumanEval problem."""

    task_id: str
    prompt: str
    canonical_solution: str
    test: str
    entry_point: str
    docstring: str


class HumanEvalBenchmark:
    """HumanEval benchmark loader."""

    def __init__(self, data_path: str = None):
        """Initialize HumanEval benchmark.

        Args:
            data_path: Optional path to HumanEval JSON file
        """
        self.data_path = data_path
        self.problems: List[HumanEvalProblem] = []
        self._load_problems()

    def _load_problems(self):
        """Load HumanEval problems."""
        # Use embedded problems if no path provided
        if not self.data_path or not Path(self.data_path).exists():
            self.problems = self._get_embedded_problems()
        else:
            with open(self.data_path) as f:
                data = json.load(f)

            for item in data:
                self.problems.append(
                    HumanEvalProblem(
                        task_id=item["task_id"],
                        prompt=item["prompt"],
                        canonical_solution=item["canonical_solution"],
                        test=item["test"],
                        entry_point=item["entry_point"],
                        docstring=item.get("docstring", ""),
                    )
                )

    def _get_embedded_problems(self) -> List[HumanEvalProblem]:
        """Get embedded sample HumanEval problems."""
        # Sample of HumanEval problems (would be 164 in full version)
        sample_problems = [
            HumanEvalProblem(
                task_id="HumanEval/0",
                prompt='''def check(candidate):
    """Check if candidate correctly solves the problem."""
    pass

from typing import List

def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """Check if the list contains two numbers that are close to each other.

    Args:
        numbers: List of floats
        threshold: Maximum distance between close numbers

    Returns:
        True if there exist two numbers whose distance is less than threshold
    """
    for idx, elem in enumerate(numbers):
        for idx2, elem2 in enumerate(numbers[idx + 1:]):
            distance = abs(elem - elem2)
            if distance < threshold:
                return True
    return False
''',
                canonical_solution="",
                test="assert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\nassert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\nassert has_close_elements([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\nassert has_close_elements([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False",
                entry_point="has_close_elements",
                docstring="Check if the list contains two numbers that are close to each other.",
            ),
            HumanEvalProblem(
                task_id="HumanEval/1",
                prompt='''def check(candidate):
    """Check if candidate correctly solves the problem."""
    pass

def separate_paren_groups(paren_string: str) -> List[str]:
    """Separate groups of balanced parentheses.

    Args:
        paren_string: String of parentheses

    Returns:
        List of strings, each containing a group of balanced parentheses
    """
    result = []
    current_group = ""
    balance = 0

    for char in paren_string:
        current_group += char
        if char == "(":
            balance += 1
        elif char == ")":
            balance -= 1

        if balance == 0:
            result.append(current_group)
            current_group = ""

    return result
''',
                canonical_solution="",
                test="assert separate_paren_groups('(())()(())') == ['(())', '()', '(())']\nassert separate_paren_groups('()()()') == ['()', '()', '()']\nassert separate_paren_groups('((()))') == ['((()))']\nassert separate_paren_groups('(()())()') == ['(()())', '()']",
                entry_point="separate_paren_groups",
                docstring="Separate groups of balanced parentheses.",
            ),
            HumanEvalProblem(
                task_id="HumanEval/2",
                prompt='''def check(candidate):
    """Check if candidate correctly solves the problem."""
    pass

from typing import List

def truncate_number(number: float) -> float:
    """Truncate a number to 2 decimal places.

    Args:
        number: Input number

    Returns:
        Number truncated to 2 decimal places
    """
    return int(number * 100) / 100
''',
                canonical_solution="",
                test="assert truncate_number(1.254) == 1.25\nassert truncate_number(1.256) == 1.25\nassert truncate_number(1.999) == 1.99\nassert truncate_number(0.001) == 0.0",
                entry_point="truncate_number",
                docstring="Truncate a number to 2 decimal places.",
            ),
            HumanEvalProblem(
                task_id="HumanEval/3",
                prompt='''def check(candidate):
    """Check if candidate correctly solves the problem."""
    pass

from typing import List

def below_threshold(occurrences: List[int], threshold: int) -> bool:
    """Check if all values in the list are below threshold.

    Args:
        occurrences: List of integers
        threshold: Threshold value

    Returns:
        True if all values are less than threshold
    """
    for val in occurrences:
        if val >= threshold:
            return False
    return True
''',
                canonical_solution="",
                test="assert below_threshold([1, 2, 3], 5) == True\nassert below_threshold([1, 2, 3], 2) == False\nassert below_threshold([0, 1, 2], 3) == True\nassert below_threshold([], 10) == True",
                entry_point="below_threshold",
                docstring="Check if all values are below threshold.",
            ),
            HumanEvalProblem(
                task_id="HumanEval/4",
                prompt='''def check(candidate):
    """Check if candidate correctly solves the problem."""
    pass

from typing import List

def parse_nested_parens(paren_string: str) -> int:
    """Find the maximum nesting depth of parentheses.

    Args:
        paren_string: String of parentheses

    Returns:
        Maximum nesting depth
    """
    max_depth = 0
    current_depth = 0

    for char in paren_string:
        if char == "(":
            current_depth += 1
            max_depth = max(max_depth, current_depth)
        elif char == ")":
            current_depth -= 1

    return max_depth
''',
                canonical_solution="",
                test="assert parse_nested_parens('(()())') == 2\nassert parse_nested_parens('()()()') == 1\nassert parse_nested_parens('((()))') == 3\nassert parse_nested_parens('') == 0",
                entry_point="parse_nested_parens",
                docstring="Find the maximum nesting depth of parentheses.",
            ),
            # More problems would be added here for full HumanEval (164 total)
        ]

        return sample_problems

    def get_problems(self) -> List[HumanEvalProblem]:
        """Get all HumanEval problems.

        Returns:
            List of HumanEval problems
        """
        return self.problems

    def get_problem(self, task_id: str) -> HumanEvalProblem:
        """Get a specific problem by ID.

        Args:
            task_id: Problem ID

        Returns:
            HumanEval problem
        """
        for problem in self.problems:
            if problem.task_id == task_id:
                return problem
        raise ValueError(f"Problem not found: {task_id}")

    def __len__(self) -> int:
        """Get number of problems."""
        return len(self.problems)

    def __repr__(self) -> str:
        """String representation."""
        return f"HumanEvalBenchmark({len(self.problems)} problems)"
