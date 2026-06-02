"""Main LeanCoder pipeline orchestrator."""

import sys
from pathlib import Path
from datetime import datetime
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config import config


def cmd_baseline(args):
    """Run baseline evaluation."""
    from eval.baseline import run_baseline

    model_cfg = config.load_yaml("config/base.yaml")
    eval_cfg = config.load_yaml("config/eval.yaml")

    checkpoint = run_baseline(
        model_config=model_cfg,
        eval_config=eval_cfg,
        registry_path=str(config.registry_path),
        hf_token=config.hf_token,
    )

    print(f"\n✓ Baseline complete: {checkpoint.id}")


def cmd_data_build(args):
    """Build dataset version."""
    from data.pipeline import DataPipeline

    data_cfg = config.load_yaml("config/data.yaml")
    pipeline = DataPipeline(data_cfg)

    sources = [
        {
            "name": args.name or "coding_qa",
            "path": args.input,
            "type": "jsonl",
            "license": "apache-2.0",
        }
    ]

    stats = pipeline.build_version(
        version=args.version,
        sources=sources,
        eval_set_path=args.eval_set,
    )

    print(f"\n✓ Dataset {args.version} built")


def cmd_train(args):
    """Run QLoRA training."""
    from train.sft_trainer import run_qlora_training

    checkpoint = run_qlora_training(
        base_model_path=args.model,
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        dataset_version=args.dataset_version,
        registry_path=str(config.registry_path),
        config_path=args.config,
    )

    print(f"\n✓ Training complete: {checkpoint.id}")


def cmd_quantize(args):
    """Quantize model."""
    from compress.quantize import QuantMethod
    from compress.compress import quantize_model

    method = QuantMethod(args.method)

    checkpoint = quantize_model(
        model_path=args.model,
        method=method,
        bits=args.bits,
        registry_path=str(config.registry_path),
    )

    print(f"\n✓ Quantization complete: {checkpoint.id}")


def cmd_promote(args):
    """Compare and promote checkpoint."""
    from registry.promotion import compare_and_promote

    result = compare_and_promote(
        checkpoint_id=args.checkpoint_id,
        registry_path=str(config.registry_path),
        primary_metric=args.metric,
    )

    if result["promoted"]:
        print(f"\n✓ Checkpoint promoted: {args.checkpoint_id}")
    else:
        print(f"\n✗ Checkpoint not promoted")


def cmd_server(args):
    """Start inference server."""
    from serve.run_server import start_server

    start_server(
        host=args.host,
        port=args.port,
        model_path=args.model,
    )


def cmd_status(args):
    """Show pipeline status."""
    from registry.manager import ModelRegistry

    registry = ModelRegistry(str(config.registry_path))

    print("=" * 60)
    print("LeanCoder Pipeline Status")
    print("=" * 60)

    best = registry.get_best()
    if best:
        print(f"\nBest Checkpoint: {best.id}")
        print(f"  Source: {best.source}")
        print(f"  Path: {best.path}")
        if best.eval:
            print(f"  Pass@1: {best.eval.pass_at_1:.2%}")
            print(f"  Size: {best.eval.size_mb:.0f} MB")
    else:
        print("\nNo best checkpoint found")

    print(f"\nTotal Checkpoints: {len(registry.checkpoints)}")

    print("\nRecent Checkpoints:")
    for cp in registry.list_checkpoints()[:5]:
        best_mark = " ★" if cp.is_best else ""
        print(f"  {cp.id}{best_mark}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="LeanCoder CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # baseline
    subparsers.add_parser("baseline", help="Run baseline evaluation")

    # data-build
    data_parser = subparsers.add_parser("data-build", help="Build dataset version")
    data_parser.add_argument("--version", required=True, help="Dataset version (e.g., ds-v1)")
    data_parser.add_argument("--input", required=True, help="Input data path")
    data_parser.add_argument("--name", help="Source name")
    data_parser.add_argument("--eval-set", help="Eval set path for leakage check")

    # train
    train_parser = subparsers.add_parser("train", help="Run QLoRA training")
    train_parser.add_argument("--model", required=True, help="Base model path")
    train_parser.add_argument("--train-data", required=True, help="Training data path")
    train_parser.add_argument("--val-data", required=True, help="Validation data path")
    train_parser.add_argument("--dataset-version", required=True, help="Dataset version")
    train_parser.add_argument("--config", default="config/train/qlora.yaml", help="Training config")

    # quantize
    quantize_parser = subparsers.add_parser("quantize", help="Quantize model")
    quantize_parser.add_argument("--model", required=True, help="Model path")
    quantize_parser.add_argument("--method", default="gguf", choices=["gguf", "awq", "gptq"], help="Quantization method")
    quantize_parser.add_argument("--bits", type=int, default=4, choices=[4, 8], help="Quantization bits")

    # promote
    promote_parser = subparsers.add_parser("promote", help="Compare and promote checkpoint")
    promote_parser.add_argument("--checkpoint-id", required=True, help="Checkpoint ID")
    promote_parser.add_argument("--metric", default="pass_at_1", help="Primary metric")

    # server
    server_parser = subparsers.add_parser("serve", help="Start inference server")
    server_parser.add_argument("--host", default="0.0.0.0", help="Host")
    server_parser.add_argument("--port", type=int, default=8000, help="Port")
    server_parser.add_argument("--model", help="Model path (default: load best from registry)")

    # status
    subparsers.add_parser("status", help="Show pipeline status")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Execute command
    commands = {
        "baseline": cmd_baseline,
        "data-build": cmd_data_build,
        "train": cmd_train,
        "quantize": cmd_quantize,
        "promote": cmd_promote,
        "serve": cmd_server,
        "status": cmd_status,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
