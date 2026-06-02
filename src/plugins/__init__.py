"""Plugin system for extending LeanCoder functionality."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from pathlib import Path
import importlib.util
import inspect


class Plugin(ABC):
    """Base plugin class."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin name."""

    @property
    @abstractmethod
    def version(self) -> str:
        """Plugin version."""

    @abstractmethod
    def initialize(self, config: Dict[str, Any] = None):
        """Initialize plugin.

        Args:
            config: Optional plugin configuration
        """

    def on_training_start(self, context: Dict):
        """Called when training starts.

        Args:
            context: Training context
        """
        pass

    def on_training_end(self, context: Dict):
        """Called when training ends.

        Args:
            context: Training context
        """
        pass

    def on_eval_start(self, context: Dict):
        """Called when evaluation starts.

        Args:
            context: Evaluation context
        """
        pass

    def on_eval_end(self, context: Dict, results: Dict):
        """Called when evaluation ends.

        Args:
            context: Evaluation context
            results: Evaluation results
        """
        pass

    def on_checkpoint_save(self, context: Dict, checkpoint_path: str):
        """Called when checkpoint is saved.

        Args:
            context: Training context
            checkpoint_path: Path to saved checkpoint
        """
        pass


class PluginManager:
    """Manage plugin lifecycle."""

    def __init__(self):
        """Initialize plugin manager."""
        self.plugins: Dict[str, Plugin] = {}
        self.plugin_configs: Dict[str, Dict] = {}

    def register_plugin(self, plugin: Plugin, config: Dict = None):
        """Register a plugin.

        Args:
            plugin: Plugin instance
            config: Optional plugin configuration
        """
        plugin_name = plugin.name

        if plugin_name in self.plugins:
            print(f"Plugin {plugin_name} already registered, skipping")
            return

        self.plugins[plugin_name] = plugin
        self.plugin_configs[plugin_name] = config or {}

        # Initialize plugin
        plugin.initialize(self.plugin_configs[plugin_name])

        print(f"Registered plugin: {plugin_name} v{plugin.version}")

    def load_plugin_from_file(self, plugin_path: str, config: Dict = None):
        """Load plugin from Python file.

        Args:
            plugin_path: Path to plugin file
            config: Optional plugin configuration
        """
        plugin_path = Path(plugin_path)

        spec = importlib.util.spec_from_file_location(
            plugin_path.stem, plugin_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Find Plugin subclass
        for name, obj in inspect.getmembers(module):
            if (
                inspect.isclass(obj)
                and issubclass(obj, Plugin)
                and obj != Plugin
            ):
                plugin_instance = obj()
                self.register_plugin(plugin_instance, config)
                return

        raise ValueError(f"No Plugin subclass found in {plugin_path}")

    def load_plugins_from_directory(self, plugins_dir: str):
        """Load all plugins from directory.

        Args:
            plugins_dir: Path to plugins directory
        """
        plugins_path = Path(plugins_dir)

        if not plugins_path.exists():
            print(f"Plugins directory not found: {plugins_dir}")
            return

        for plugin_file in plugins_path.glob("*_plugin.py"):
            try:
                self.load_plugin_from_file(plugin_file)
            except Exception as e:
                print(f"Failed to load plugin {plugin_file}: {e}")

    def get_plugin(self, plugin_name: str) -> Optional[Plugin]:
        """Get registered plugin.

        Args:
            plugin_name: Name of plugin

        Returns:
            Plugin or None
        """
        return self.plugins.get(plugin_name)

    def trigger_hook(self, hook_name: str, **kwargs):
        """Trigger a hook on all plugins.

        Args:
            hook_name: Name of hook method
            **kwargs: Arguments to pass to hook
        """
        for plugin in self.plugins.values():
            hook_method = getattr(plugin, hook_name, None)
            if hook_method and callable(hook_method):
                try:
                    hook_method(**kwargs)
                except Exception as e:
                    print(f"Plugin {plugin.name} hook {hook_name} failed: {e}")

    def __len__(self) -> int:
        """Get number of plugins."""
        return len(self.plugins)


# Built-in plugins
class WeightsAndBiasesPlugin(Plugin):
    """Weights & Biases experiment tracking plugin."""

    @property
    def name(self) -> str:
        return "wandb"

    @property
    def version(self) -> str:
        return "1.0.0"

    def initialize(self, config: Dict[str, Any] = None):
        """Initialize W&B.

        Args:
            config: W&B configuration (project, entity, etc.)
        """
        try:
            import wandb

            wandb_config = config or {}
            wandb.init(
                project=wandb_config.get("project", "leancoder"),
                entity=wandb_config.get("entity"),
                name=wandb_config.get("run_name"),
                config=wandb_config.get("config", {}),
            )

            self.wandb = wandb
            print("W&B plugin initialized")

        except ImportError:
            print("W&B not installed, skipping plugin")
            self.wandb = None

    def on_training_start(self, context: Dict):
        """Log training start.

        Args:
            context: Training context
        """
        if self.wandb:
            self.wandb.config.update(context)

    def on_training_end(self, context: Dict):
        """Log training end.

        Args:
            context: Training context
        """
        if self.wandb:
            self.wandb.finish()

    def on_eval_end(self, context: Dict, results: Dict):
        """Log evaluation results.

        Args:
            context: Evaluation context
            results: Evaluation results
        """
        if self.wandb:
            self.wandb.log(results)


class SlackNotifierPlugin(Plugin):
    """Slack notification plugin."""

    @property
    def name(self) -> str:
        return "slack_notifier"

    @property
    def version(self) -> str:
        return "1.0.0"

    def initialize(self, config: Dict[str, Any] = None):
        """Initialize Slack notifier.

        Args:
            config: Slack webhook URL
        """
        self.webhook_url = config.get("webhook_url") if config else None

    def on_checkpoint_save(self, context: Dict, checkpoint_path: str):
        """Notify when checkpoint saved.

        Args:
            context: Training context
            checkpoint_path: Path to checkpoint
        """
        if self.webhook_url:
            self._send_notification(
                f"Checkpoint saved: {checkpoint_path}",
                context,
            )

    def _send_notification(self, message: str, context: Dict):
        """Send Slack notification.

        Args:
            message: Notification message
            context: Additional context
        """
        # Placeholder: would send to Slack webhook
        print(f"[Slack] {message}")


class MetricsExporterPlugin(Plugin):
    """Export metrics to various formats."""

    @property
    def name(self) -> str:
        return "metrics_exporter"

    @property
    def version(self) -> str:
        return "1.0.0"

    def initialize(self, config: Dict[str, Any] = None):
        """Initialize metrics exporter.

        Args:
            config: Export configuration (formats, output_dir)
        """
        self.formats = config.get("formats", ["json", "csv"]) if config else ["json"]
        self.output_dir = config.get("output_dir", "outputs/exports") if config else "outputs/exports"

    def on_eval_end(self, context: Dict, results: Dict):
        """Export evaluation results.

        Args:
            context: Evaluation context
            results: Evaluation results
        """
        import json
        from pathlib import Path

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        timestamp = context.get("timestamp", "now")

        if "json" in self.formats:
            json_path = Path(self.output_dir) / f"eval_{timestamp}.json"
            with open(json_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Exported metrics to {json_path}")

        if "csv" in self.formats:
            # Would convert to CSV
            pass


def create_default_plugins(config: Dict = None) -> PluginManager:
    """Create plugin manager with default plugins.

    Args:
        config: Plugin configuration

    Returns:
        PluginManager instance
    """
    manager = PluginManager()

    config = config or {}

    # Register built-in plugins if enabled
    if config.get("wandb", {}).get("enabled", False):
        manager.register_plugin(WeightsAndBiasesPlugin(), config["wandb"])

    if config.get("slack", {}).get("enabled", False):
        manager.register_plugin(SlackNotifierPlugin(), config["slack"])

    if config.get("exporter", {}).get("enabled", False):
        manager.register_plugin(MetricsExporterPlugin(), config["exporter"])

    return manager
