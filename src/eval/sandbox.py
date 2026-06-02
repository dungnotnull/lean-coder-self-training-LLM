"""Docker-based isolated execution environment for code evaluation."""

import subprocess
import tempfile
import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
import time


@dataclass
class SandboxResult:
    """Result from sandboxed execution."""

    success: bool
    output: str
    error: str
    exit_code: int
    execution_time: float


class DockerSandbox:
    """Execute code in isolated Docker container."""

    def __init__(
        self,
        image: str = "python:3.11-slim",
        timeout: int = 10,
        memory_mb: int = 512,
        network_disabled: bool = True,
    ):
        """Initialize Docker sandbox.

        Args:
            image: Docker image to use
            timeout: Execution timeout in seconds
            memory_mb: Memory limit in MB
            network_disabled: Disable network access
        """
        self.image = image
        self.timeout = timeout
        self.memory_mb = memory_mb
        self.network_disabled = network_disabled

    def execute(
        self,
        code: str,
        tests: list[str] = None,
        imports: list[str] = None,
    ) -> SandboxResult:
        """Execute code in isolated Docker container.

        Args:
            code: Python code to execute
            tests: Optional list of test assertions
            imports: Optional list of allowed imports

        Returns:
            SandboxResult with execution outcome
        """
        # Build complete script
        script_parts = []

        # Add allowed imports if specified
        if imports:
            script_parts.append("# Allowed imports")
            for imp in imports:
                script_parts.append(f"import {imp}")
            script_parts.append("")

        # Add code
        script_parts.append("# User code")
        script_parts.append(code)
        script_parts.append("")

        # Add tests if specified
        if tests:
            script_parts.append("# Tests")
            script_parts.append("test_results = []")
            for i, test in enumerate(tests):
                script_parts.append(f"try:")
                script_parts.append(f"    {test}")
                script_parts.append(f"    test_results.append(True)")
                script_parts.append(f"except Exception as e:")
                script_parts.append(f"    test_results.append(False)")
                script_parts.append(f"    print(f'Test {i} failed: {{e}}')")

            script_parts.append("")
            script_parts.append("# Report results")
            script_parts.append("import sys")
            script_parts.append("if all(test_results):")
            script_parts.append("    print('SUCCESS: All tests passed')")
            script_parts.append("    sys.exit(0)")
            script_parts.append("else:")
            script_parts.append("    print(f'FAILURE: {sum(test_results)}/{len(test_results)} tests passed')")
            script_parts.append("    sys.exit(1)")

        script = "\n".join(script_parts)

        # Write script to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(script)
            script_path = f.name

        try:
            # Build Docker command
            docker_cmd = [
                "docker",
                "run",
                "--rm",
                f"--memory={self.memory_mb}m",
                f"--memory-swap={self.memory_mb}m",
                "--cpus=1",
                "--network=none" if self.network_disabled else "--network=bridge",
                "-v", f"{Path(script_path).parent}:/workspace:ro",
                "-w", "/workspace",
                self.image,
                "timeout",
                str(self.timeout),
                "python",
                Path(script_path).name,
            ]

            # Execute
            start_time = time.time()
            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout + 5,
            )
            elapsed = time.time() - start_time

            # Check for success
            success = result.returncode == 0

            # If tests were specified, check for success marker
            if tests and result.stdout:
                success = "SUCCESS: All tests passed" in result.stdout

            return SandboxResult(
                success=success,
                output=result.stdout,
                error=result.stderr,
                exit_code=result.returncode,
                execution_time=elapsed,
            )

        except subprocess.TimeoutExpired:
            return SandboxResult(
                success=False,
                output="",
                error="Execution timeout",
                exit_code=-1,
                execution_time=self.timeout,
            )
        except FileNotFoundError:
            # Docker not available, fall back to subprocess
            return self._fallback_execute(code, tests)
        finally:
            # Clean up temp file
            try:
                Path(script_path).unlink()
            except Exception:
                pass

    def _fallback_execute(self, code: str, tests: list[str] = None) -> SandboxResult:
        """Fallback to subprocess execution when Docker unavailable.

        Args:
            code: Code to execute
            tests: Optional tests

        Returns:
            SandboxResult
        """
        # Use the original process-based sandbox
        from .harness import EvalHarness

        harness = EvalHarness({
            "sandbox": {
                "timeout_seconds": self.timeout,
                "memory_limit_mb": self.memory_mb,
            }
        })

        # Build complete code
        full_code = code
        if tests:
            full_code += "\n\n# Tests\n"
            for test in tests:
                full_code += f"{test}\n"

        # Execute
        success = harness._execute_sandboxed(full_code, tests or [])

        return SandboxResult(
            success=success,
            output="Executed via fallback" if success else "Tests failed",
            error="",
            exit_code=0 if success else 1,
            execution_time=0.0,
        )

    def check_docker_available(self) -> bool:
        """Check if Docker is available.

        Returns:
            True if Docker is installed and running
        """
        try:
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except Exception:
            return False


class MultiTestSandbox:
    """Run multiple test cases efficiently."""

    def __init__(self, sandbox: DockerSandbox):
        """Initialize multi-test sandbox.

        Args:
            sandbox: Docker sandbox instance
        """
        self.sandbox = sandbox

    def run_test_suite(
        self,
        code: str,
        test_cases: list[dict],
    ) -> dict:
        """Run a suite of test cases.

        Args:
            code: Code to test
            test_cases: List of test cases with input and expected output

        Returns:
            Dict with test results
        """
        results = {
            "total": len(test_cases),
            "passed": 0,
            "failed": 0,
            "details": [],
        }

        for i, test in enumerate(test_cases):
            # Build test code
            test_code = f"{code}\n\n# Test case {i}\n"
            test_code += f"input_data = {test.get('input')}\n"
            test_code += f"expected = {test.get('expected')}\n"
            test_code += "result = solve(input_data) if 'solve' in dir() else None\n"
            test_code += "assert result == expected, f'Expected {expected}, got {result}'\n"
            test_code += "print('PASS')\n"

            result = self.sandbox.execute(test_code)

            passed = result.success and "PASS" in result.output
            if passed:
                results["passed"] += 1
            else:
                results["failed"] += 1

            results["details"].append({
                "test_id": i,
                "passed": passed,
                "output": result.output,
                "error": result.error,
            })

        results["pass_rate"] = results["passed"] / results["total"] if results["total"] > 0 else 0

        return results
