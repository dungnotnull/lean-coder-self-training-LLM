"""Evaluation report generation."""

from datetime import datetime
from pathlib import Path
from typing import Optional


class EvalReporter:
    """Writes timestamped eval reports."""

    def __init__(self, outputs_dir: str):
        """Initialize reporter with outputs directory.

        Args:
            outputs_dir: Directory for output reports
        """
        self.outputs_dir = Path(outputs_dir)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

    def write_report(
        self, result, checkpoint_id: str, comparison: Optional[dict] = None
    ) -> Path:
        """Write eval report to outputs/.

        Args:
            result: EvalResult from harness
            checkpoint_id: ID of evaluated checkpoint
            comparison: Optional dict with vs_best and vs_base deltas

        Returns:
            Path to written report
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{timestamp}_{checkpoint_id}_eval.md"
        report_path = self.outputs_dir / filename

        content = self._format_report(result, checkpoint_id, comparison)

        with open(report_path, "w") as f:
            f.write(content)

        return report_path

    def _format_report(self, result, checkpoint_id: str, comparison: Optional[dict]) -> str:
        """Format report as markdown.

        Args:
            result: EvalResult
            checkpoint_id: Checkpoint ID
            comparison: Comparison deltas

        Returns:
            Formatted markdown string
        """
        lines = [
            f"# Evaluation Report — {checkpoint_id}",
            "",
            f"**Checkpoint:** {checkpoint_id}",
            f"**Suite:** {result.suite}",
            "",
            "## Results",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Pass@1 | {result.pass_at_1:.2%} |",
        ]

        # Add pass@k values
        for i, k_val in enumerate(result.pass_at_k, start=1):
            lines.append(f"| Pass@{i} | {k_val:.2%} |")

        lines.extend([
            f"| Latency | {result.latency_tok_s:.2f} tok/s |",
            f"| Size | {result.model_size_mb:.0f} MB |",
            f"| Memory | {result.memory_mb:.0f} MB |",
            "",
            "## Comparison",
            "",
        ])

        if comparison:
            lines.append("| vs. | Pass@1 Delta | Pass@10 Delta | Size Delta |")
            lines.append("|-----|--------------|---------------|------------|")

            for name, delta in comparison.items():
                pass1_delta = delta.get("pass_at_1", 0)
                pass10_delta = delta.get("pass_at_10", 0)
                size_delta = delta.get("size_mb", 0)

                lines.append(
                    f"| {name} | {pass1_delta:+.2%} | {pass10_delta:+.2%} | {size_delta:+.0f} MB |"
                )
        else:
            lines.append("No comparison data available.")

        lines.extend([
            "",
            "## Details",
            "",
            f"- Total problems: {result.problems_total}",
            f"- Samples per problem: {result.details[0]['n'] if result.details else 'N/A'}",
            "",
            "## Problem Results",
            "",
            "| Problem ID | Passed | Samples |",
            "|------------|--------|---------|",
        ])

        for detail in result.details[:50]:  # Limit to first 50
            lines.append(
                f"| {detail['problem_id']} | {detail['c']} | {detail['n']} |"
            )

        if len(result.details) > 50:
            lines.append(f"| ... | ... | ... |")
            lines.append(f"| **Total** | **{sum(d['c'] for d in result.details)}** | **{len(result.details) * (result.details[0]['n'] if result.details else 0)}** |")

        return "\n".join(lines)
