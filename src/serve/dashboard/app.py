"""FastAPI dashboard application."""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from typing import Dict
from pathlib import Path
import json


def create_dashboard_app(
    metrics_path: str = None,
    registry_path: str = None,
    brain_path: str = None,
) -> FastAPI:
    """Create dashboard FastAPI app.

    Args:
        metrics_path: Path to metrics storage
        registry_path: Path to model registry
        brain_path: Path to knowledge brain

    Returns:
        FastAPI application
    """
    app = FastAPI(
        title="LeanCoder Dashboard",
        description="Real-time training and evaluation monitoring",
        version="1.0.0",
    )

    # Initialize components
    metrics_collector = None
    registry_view = None
    brain_view = None

    if metrics_path:
        from .metrics import MetricsCollector
        metrics_collector = MetricsCollector(metrics_path)

    if registry_path:
        from .registry import RegistryView
        registry_view = RegistryView(registry_path)

    if brain_path:
        from .registry import KnowledgeBrainView
        brain_view = KnowledgeBrainView(brain_path)

    @app.get("/", response_class=HTMLResponse)
    async def root():
        """Dashboard home page."""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>LeanCoder Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #e2e8f0; }
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 12px; margin-bottom: 30px; }
        .header h1 { font-size: 2.5rem; color: white; margin-bottom: 10px; }
        .header p { color: rgba(255,255,255,0.8); }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .card { background: #1e293b; border-radius: 12px; padding: 20px; border: 1px solid #334155; }
        .card h2 { font-size: 1.25rem; margin-bottom: 15px; color: #60a5fa; }
        .metric { font-size: 2.5rem; font-weight: bold; color: #22d3ee; }
        .metric-label { color: #94a3b8; font-size: 0.875rem; }
        .status { display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 0.875rem; }
        .status.running { background: #22c55e; color: white; }
        .status.idle { background: #64748b; color: white; }
        .table { width: 100%; border-collapse: collapse; margin-top: 10px; }
        .table th { text-align: left; padding: 12px; border-bottom: 1px solid #334155; color: #94a3b8; }
        .table td { padding: 12px; border-bottom: 1px solid #1e293b; }
        .badge { display: inline-block; padding: 4px 8px; border-radius: 4px; font-size: 0.75rem; }
        .badge.best { background: #22c55e; color: white; }
        .badge.normal { background: #3b82f6; color: white; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>LeanCoder Dashboard</h1>
            <p>Real-time training and evaluation monitoring</p>
        </div>

        <div class="grid">
            <div class="card">
                <h2>Training Status</h2>
                <span class="status running">Active</span>
                <div style="margin-top: 15px;">
                    <span class="metric">0</span>
                    <span class="metric-label">Current Step</span>
                </div>
            </div>

            <div class="card">
                <h2>Loss</h2>
                <span class="metric" id="loss">0.00</span>
                <span class="metric-label">Latest Loss</span>
            </div>

            <div class="card">
                <h2>Eval Score</h2>
                <span class="metric" id="eval">0.0%</span>
                <span class="metric-label">Pass@1</span>
            </div>

            <div class="card">
                <h2>Learning Rate</h2>
                <span class="metric" id="lr">0.00</span>
                <span class="metric-label">Current LR</span>
            </div>
        </div>

        <div class="card">
            <h2>Model Registry</h2>
            <div id="registry">
                <p>Loading...</p>
            </div>
        </div>

        <div class="card" style="margin-top: 20px;">
            <h2>Knowledge Brain</h2>
            <div id="brain">
                <p>Loading...</p>
            </div>
        </div>
    </div>

    <script>
        // Load registry
        fetch('/api/registry/summary')
            .then(r => r.json())
            .then(data => {
                document.getElementById('registry').innerHTML = `
                    <table class="table">
                        <tr><th>Checkpoint</th><th>Source</th><th>Pass@1</th><th>Status</th></tr>
                        ${data.recent_checkpoints.map(cp => `
                            <tr>
                                <td>${cp.id}</td>
                                <td>${cp.source}</td>
                                <td>${(cp.pass_at_1 * 100).toFixed(1)}%</td>
                                <td>${cp.id === data.best_checkpoint ? '<span class="badge best">Best</span>' : '<span class="badge normal">Checkpoint</span>'}</td>
                            </tr>
                        `).join('')}
                    </table>
                `;
            });

        // Load brain
        fetch('/api/brain/summary')
            .then(r => r.json())
            .then(data => {
                document.getElementById('brain').innerHTML = `
                    <p><strong>Version:</strong> ${data.current_version}</p>
                    <p><strong>Iterations Promoted:</strong> ${data.iterations_promoted}</p>
                    <p><strong>Best Checkpoint:</strong> ${data.best_checkpoint || 'None'}</p>
                `;
            });

        // Auto-refresh every 5 seconds
        setInterval(() => {
            fetch('/api/registry/summary').then(r => r.json()).then(data => {
                // Update display
            });
        }, 5000);
    </script>
</body>
</html>
        """

    @app.get("/api/health")
    async def health():
        """Health check endpoint."""
        return {"status": "healthy", "timestamp": "2026-06-02"}

    @app.get("/api/metrics/summary")
    async def metrics_summary():
        """Get metrics summary."""
        if metrics_collector:
            return metrics_collector.get_training_summary()
        return {"error": "Metrics not available"}

    @app.get("/api/metrics/{metric_name}")
    async def get_metrics(metric_name: str):
        """Get specific metric."""
        if metrics_collector:
            points = metrics_collector.get_metrics(metric_name)
            return {"metric_name": metric_name, "points": points}
        return {"error": "Metrics not available"}

    @app.get("/api/registry/summary")
    async def registry_summary():
        """Get registry summary."""
        if registry_view:
            return registry_view.get_summary()
        return {"error": "Registry not available"}

    @app.get("/api/registry/checkpoint/{checkpoint_id}")
    async def get_checkpoint(checkpoint_id: str):
        """Get checkpoint details."""
        if registry_view:
            return registry_view.get_checkpoint(checkpoint_id)
        return {"error": "Registry not available"}

    @app.get("/api/registry/history")
    async def registry_history():
        """Get checkpoint history."""
        if registry_view:
            return {"checkpoints": registry_view.get_history()}
        return {"error": "Registry not available"}

    @app.get("/api/brain/summary")
    async def brain_summary():
        """Get brain summary."""
        if brain_view:
            return brain_view.get_summary()
        return {"error": "Brain not available"}

    @app.get("/api/brain/timeline")
    async def brain_timeline():
        """Get improvement timeline."""
        if brain_view:
            return {"entries": brain_view.get_timeline()}
        return {"error": "Brain not available"}

    return app


if __name__ == "__main__":
    import uvicorn
    from config import config

    app = create_dashboard_app(
        metrics_path=str(config.outputs_dir / "metrics.json"),
        registry_path=str(config.registry_path),
        brain_path=str(config.root_dir / "knowledge" / "brain.json"),
    )

    uvicorn.run(app, host="0.0.0.0", port=8001)
