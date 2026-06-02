"""Metrics collection for dashboard."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
import json
from pathlib import Path


@dataclass
class MetricPoint:
    """A single metric data point."""

    timestamp: str
    step: int
    value: float
    metric_name: str


class MetricsCollector:
    """Collect and store training metrics."""

    def __init__(self, storage_path: str = None):
        """Initialize metrics collector.

        Args:
            storage_path: Path to store metrics JSON
        """
        self.storage_path = Path(storage_path) if storage_path else None
        self.metrics: Dict[str, List[MetricPoint]] = {}

        if self.storage_path:
            self._load_metrics()

    def _load_metrics(self):
        """Load metrics from storage."""
        if not self.storage_path.exists():
            return

        try:
            with open(self.storage_path) as f:
                data = json.load(f)

            for metric_name, points in data.items():
                self.metrics[metric_name] = [
                    MetricPoint(**point) for point in points
                ]
        except Exception as e:
            print(f"Failed to load metrics: {e}")

    def record(self, metric_name: str, value: float, step: int = None):
        """Record a metric value.

        Args:
            metric_name: Name of the metric
            value: Metric value
            step: Optional step number
        """
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []

        point = MetricPoint(
            timestamp=datetime.now().isoformat(),
            step=step if step is not None else len(self.metrics[metric_name]),
            value=value,
            metric_name=metric_name,
        )

        self.metrics[metric_name].append(point)

        if self.storage_path:
            self._save_metrics()

    def get_metrics(self, metric_name: str) -> List[MetricPoint]:
        """Get all points for a metric.

        Args:
            metric_name: Name of the metric

        Returns:
            List of metric points
        """
        return self.metrics.get(metric_name, [])

    def get_all_metrics(self) -> Dict[str, List[MetricPoint]]:
        """Get all metrics.

        Returns:
            Dict mapping metric names to points
        """
        return self.metrics

    def get_latest(self, metric_name: str) -> Optional[MetricPoint]:
        """Get latest value for a metric.

        Args:
            metric_name: Name of the metric

        Returns:
            Latest metric point or None
        """
        points = self.metrics.get(metric_name, [])
        return points[-1] if points else None

    def _save_metrics(self):
        """Save metrics to storage."""
        if not self.storage_path:
            return

        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        data = {}
        for metric_name, points in self.metrics.items():
            data[metric_name] = [
                {
                    "timestamp": p.timestamp,
                    "step": p.step,
                    "value": p.value,
                    "metric_name": p.metric_name,
                }
                for p in points
            ]

        with open(self.storage_path, "w") as f:
            json.dump(data, f, indent=2)


class TrainingMetrics(MetricsCollector):
    """Specialized metrics collector for training."""

    def record_loss(self, loss: float, step: int):
        """Record training loss.

        Args:
            loss: Loss value
            step: Training step
        """
        self.record("train_loss", loss, step)

    def record_eval_score(self, score: float, step: int):
        """Record evaluation score.

        Args:
            score: Evaluation score (pass@1, etc.)
            step: Training step
        """
        self.record("eval_pass_at_1", score, step)

    def record_lr(self, lr: float, step: int):
        """Record learning rate.

        Args:
            lr: Learning rate
            step: Training step
        """
        self.record("learning_rate", lr, step)

    def record_gradient_norm(self, norm: float, step: int):
        """Record gradient norm.

        Args:
            norm: Gradient norm value
            step: Training step
        """
        self.record("gradient_norm", norm, step)

    def get_training_summary(self) -> Dict:
        """Get summary of training metrics.

        Returns:
            Summary dictionary
        """
        latest_loss = self.get_latest("train_loss")
        latest_eval = self.get_latest("eval_pass_at_1")
        latest_lr = self.get_latest("learning_rate")

        return {
            "latest_loss": latest_loss.value if latest_loss else 0.0,
            "latest_eval_score": latest_eval.value if latest_eval else 0.0,
            "current_lr": latest_lr.value if latest_lr else 0.0,
            "total_steps": len(self.metrics.get("train_loss", [])),
            "eval_steps": len(self.metrics.get("eval_pass_at_1", [])),
        }
