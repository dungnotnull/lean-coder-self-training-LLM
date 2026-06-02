"""Inference server startup script."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from serve.server import InferenceServer, ServerConfig
from serve.api import create_app
from registry.manager import ModelRegistry
from config import config
import uvicorn


def start_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    model_path: str = None,
):
    """Start the inference server.

    Args:
        host: Host to bind to
        port: Port to bind to
        model_path: Optional model path override
    """
    print("=" * 60)
    print("LeanCoder Inference Server")
    print("=" * 60)

    # Load best model from registry if not specified
    if not model_path:
        registry = ModelRegistry(str(config.registry_path))
        best = registry.get_best()

        if best:
            model_path = best.path
            print(f"Loading best checkpoint: {best.id}")
        else:
            print("No best checkpoint found in registry")
            return

    # Create server config
    server_config = ServerConfig(
        host=host,
        port=port,
        model_path=model_path,
    )

    # Create server instance
    server = InferenceServer(server_config)

    # Load model
    print(f"\nLoading model from: {model_path}")
    server.load_model()

    # Create FastAPI app
    app = create_app(server)

    print(f"\nStarting server on {host}:{port}")
    print("=" * 60)

    # Run server
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_server(
        host="0.0.0.0",
        port=8000,
    )
