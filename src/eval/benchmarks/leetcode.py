"""LeetCode-style coding problems."""

from typing import List, Dict
from dataclasses import dataclass


@dataclass
class LeetCodeProblem:
    """A LeetCode-style problem."""

    problem_id: str
    title: str
    difficulty: str  # Easy, Medium, Hard
    description: str
    examples: List[Dict]
    constraints: List[str]
    starter_code: str
    test_cases: List[Dict]


class LeetCodeBenchmark:
    """LeetCode-style benchmark loader."""

    def __init__(self, data_path: str = None):
        """Initialize LeetCode benchmark.

        Args:
            data_path: Optional path to problems JSON file
        """
        self.data_path = data_path
        self.problems: List[LeetCodeProblem] = []
        self._load_problems()

    def _load_problems(self):
        """Load LeetCode problems."""
        if not self.data_path or not Path(self.data_path).exists():
            self.problems = self._get_embedded_problems()
        else:
            import json

            with open(self.data_path) as f:
                data = json.load(f)

            for item in data:
                self.problems.append(
                    LeetCodeProblem(
                        problem_id=item["problem_id"],
                        title=item["title"],
                        difficulty=item["difficulty"],
                        description=item["description"],
                        examples=item.get("examples", []),
                        constraints=item.get("constraints", []),
                        starter_code=item.get("starter_code", ""),
                        test_cases=item.get("test_cases", []),
                    )
                )

    def _get_embedded_problems(self) -> List[LeetCodeProblem]:
        """Get embedded sample problems."""
        sample_problems = [
            LeetCodeProblem(
                problem_id="LC-1",
                title="Two Sum",
                difficulty="Easy",
                description="Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.",
                examples=[
                    {
                        "input": "nums = [2,7,11,15], target = 9",
                        "output": "[0,1]",
                        "explanation": "Because nums[0] + nums[1] == 9, we return [0, 1].",
                    }
                ],
                constraints=[
                    "2 <= nums.length <= 10^4",
                    "-10^9 <= nums[i] <= 10^9",
                    "-10^9 <= target <= 10^9",
                ],
                starter_code="""
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        pass
""",
                test_cases=[
                    {"input": {"nums": [2, 7, 11, 15], "target": 9}, "expected": [0, 1]},
                    {"input": {"nums": [3, 2, 4], "target": 6}, "expected": [1, 2]},
                    {"input": {"nums": [3, 3], "target": 6}, "expected": [0, 1]},
                ],
            ),
            LeetCodeProblem(
                problem_id="LC-2",
                title="Add Two Numbers",
                difficulty="Medium",
                description="You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order, and each of their nodes contains a single digit. Add the two numbers and return the sum as a linked list.",
                examples=[
                    {
                        "input": "l1 = [2,4,3], l2 = [5,6,4]",
                        "output": "[7,0,8]",
                        "explanation": "342 + 465 = 807.",
                    }
                ],
                constraints=[
                    "The number of nodes in each linked list is in the range [1, 100].",
                    "0 <= Node.val <= 9",
                ],
                starter_code="""
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        pass
""",
                test_cases=[
                    {
                        "input": {"l1": [2, 4, 3], "l2": [5, 6, 4]},
                        "expected": [7, 0, 8],
                    }
                ],
            ),
            LeetCodeProblem(
                problem_id="LC-3",
                title="Longest Substring Without Repeating Characters",
                difficulty="Medium",
                description="Given a string s, find the length of the longest substring without repeating characters.",
                examples=[
                    {"input": "s = 'abcabcbb'", "output": "4", "explanation": "The answer is 'abc', with length 3."},
                    {"input": "s = 'bbbbb'", "output": "1", "explanation": "The answer is 'b', with length 1."},
                    {"input": "s = 'pwwkew'", "output": "3", "explanation": "The answer is 'wke', with length 3."},
                ],
                constraints=[
                    "0 <= s.length <= 5 * 10^4",
                    "s consists of English letters, digits, symbols and spaces.",
                ],
                starter_code="""
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        pass
""",
                test_cases=[
                    {"input": {"s": "abcabcbb"}, "expected": 3},
                    {"input": {"s": "bbbbb"}, "expected": 1},
                    {"input": {"s": "pwwkew"}, "expected": 3},
                ],
            ),
            LeetCodeProblem(
                problem_id="LC-4",
                title="Valid Parentheses",
                difficulty="Easy",
                description="Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.",
                examples=[
                    {"input": "s = '()'", "output": "true"},
                    {"input": "s = '()[]{}'", "output": "true"},
                    {"input": "s = '(]'", "output": "false"},
                ],
                constraints=[
                    "1 <= s.length <= 10^4",
                    "s consists of parentheses only '()[]{}'.",
                ],
                starter_code="""
class Solution:
    def isValid(self, s: str) -> bool:
        pass
""",
                test_cases=[
                    {"input": {"s": "()"}, "expected": True},
                    {"input": {"s": "()[]{}"}, "expected": True},
                    {"input": {"s": "(]"}, "expected": False},
                    {"input": {"s": "([)]"}, "expected": False},
                    {"input": {"s": "{[]}"}, "expected": True},
                ],
            ),
            LeetCodeProblem(
                problem_id="LC-5",
                title="Merge Two Sorted Lists",
                difficulty="Easy",
                description="You are given the heads of two sorted linked lists. Merge the two lists in a one sorted list and return.",
                examples=[
                    {"input": "list1 = [1,2,4], list2 = [1,3,4]", "output": "[1,1,2,3,4,4]"},
                ],
                constraints=[
                    "The number of nodes in both lists is in the range [0, 50].",
                    "-100 <= Node.val <= 100",
                ],
                starter_code="""
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def mergeTwoLists(self, list1: ListNode, list2: ListNode) -> ListNode:
        pass
""",
                test_cases=[
                    {"input": {"list1": [1, 2, 4], "list2": [1, 3, 4]}, "expected": [1, 1, 2, 3, 4, 4]},
                    {"input": {"list1": [], "list2": []}, "expected": []},
                    {"input": {"list1": [], "list2": [0]}, "expected": [0]},
                ],
            ),
        ]

        return sample_problems

    def get_problems(self) -> List[LeetCodeProblem]:
        """Get all LeetCode problems.

        Returns:
            List of LeetCode problems
        """
        return self.problems

    def get_problems_by_difficulty(self, difficulty: str) -> List[LeetCodeProblem]:
        """Get problems filtered by difficulty.

        Args:
            difficulty: Easy, Medium, or Hard

        Returns:
            List of problems
        """
        return [p for p in self.problems if p.difficulty == difficulty]

    def get_problem(self, problem_id: str) -> LeetCodeProblem:
        """Get a specific problem by ID.

        Args:
            problem_id: Problem ID

        Returns:
            LeetCode problem
        """
        for problem in self.problems:
            if problem.problem_id == problem_id:
                return problem
        raise ValueError(f"Problem not found: {problem_id}")

    def __len__(self) -> int:
        """Get number of problems."""
        return len(self.problems)

    def __repr__(self) -> str:
        """String representation."""
        return f"LeetCodeBenchmark({len(self.problems)} problems)"
