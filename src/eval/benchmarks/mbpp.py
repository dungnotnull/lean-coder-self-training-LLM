"""MBPP (Mostly Basic Python Problems) benchmark - 974 problems."""

from typing import List
from dataclasses import dataclass


@dataclass
class MBPPProblem:
    """An MBPP problem."""

    task_id: int
    text: str
    code: str
    test_list: List[str]
    test_setup: int
    challenge_list: List[str]


class MBPPBenchmark:
    """MBPP benchmark loader."""

    def __init__(self, data_path: str = None):
        """Initialize MBPP benchmark.

        Args:
            data_path: Optional path to MBPP JSON file
        """
        self.data_path = data_path
        self.problems: List[MBPPProblem] = []
        self._load_problems()

    def _load_problems(self):
        """Load MBPP problems."""
        if not self.data_path or not Path(self.data_path).exists():
            self.problems = self._get_embedded_problems()
        else:
            import json

            with open(self.data_path) as f:
                data = json.load(f)

            for i, item in enumerate(data):
                self.problems.append(
                    MBPPProblem(
                        task_id=i,
                        text=item["text"],
                        code=item["code"],
                        test_list=item["test_list"],
                        test_setup=item.get("test_setup", 0),
                        challenge_list=item.get("challenge_list", []),
                    )
                )

    def _get_embedded_problems(self) -> List[MBPPProblem]:
        """Get embedded sample MBPP problems."""
        # Sample of MBPP problems (would be 974 in full version)
        sample_problems = [
            MBPPProblem(
                task_id=0,
                text="Write a function to find the minimum number of rotations required to obtain the original string.",
                code="""
def min_rotations_to_original(s):
    n = len(s)
    temp = s + s
    for i in range(1, n + 1):
        if temp[i:i+n] == s:
            return i
    return n
""",
                test_list=[
                    "assert min_rotations_to_original('aaa') == 1",
                    "assert min_rotations_to_original('abcd') == 4",
                    "assert min_rotations_to_original('abab') == 2",
                ],
                test_setup=0,
                challenge_list=[],
            ),
            MBPPProblem(
                task_id=1,
                text="Write a function to find the minimum element in a rated and sorted list.",
                code="""
def min_rated_sorted_list(lst):
    return min(lst[0]) if lst else None
""",
                test_list=[
                    "assert min_rated_sorted_list([[4, 2, 3], [1, 2, 3], [5, 6, 7]]) == 1",
                    "assert min_rated_sorted_list([[10, 20, 30], [40, 50, 60]]) == 10",
                ],
                test_setup=0,
                challenge_list=[],
            ),
            MBPPProblem(
                task_id=2,
                text="Write a function to find the list of lists with minimum length.",
                code="""
def min_length_list(lst):
    min_len = min(len(sublist) for sublist in lst)
    return [sublist for sublist in lst if len(sublist) == min_len]
""",
                test_list=[
                    "assert min_length_list([[1], [1, 2], [1, 2, 3]]) == [[1]]",
                    "assert min_length_list([[1, 2], [1, 2, 3], [1, 2, 3, 4]]) == [[1, 2]]",
                ],
                test_setup=0,
                challenge_list=[],
            ),
            MBPPProblem(
                task_id=3,
                text="Write a function to find the maximum of similar elements in two lists.",
                code="""
def max_similar_elements(list1, list2):
    common = set(list1) & set(list2)
    return max(common) if common else None
""",
                test_list=[
                    "assert max_similar_elements([1, 2, 3, 4], [2, 3, 4, 5]) == 4",
                    "assert max_similar_elements([1, 2, 3], [4, 5, 6]) == None",
                ],
                test_setup=0,
                challenge_list=[],
            ),
            MBPPProblem(
                task_id=4,
                text="Write a function to find the second smallest number in a list.",
                code="""
def second_smallest(numbers):
    if len(numbers) < 2:
        return None
    unique = sorted(set(numbers))
    return unique[1] if len(unique) > 1 else None
""",
                test_list=[
                    "assert second_smallest([1, 2, 3, 4, 5]) == 2",
                    "assert second_smallest([5, 4, 3, 2, 1]) == 2",
                    "assert second_smallest([1, 1, 2]) == 2",
                ],
                test_setup=0,
                challenge_list=[],
            ),
        ]

        return sample_problems

    def get_problems(self) -> List[MBPPProblem]:
        """Get all MBPP problems.

        Returns:
            List of MBPP problems
        """
        return self.problems

    def get_problem(self, task_id: int) -> MBPPProblem:
        """Get a specific problem by ID.

        Args:
            task_id: Problem ID

        Returns:
            MBPP problem
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
        return f"MBPPBenchmark({len(self.problems)} problems)"
