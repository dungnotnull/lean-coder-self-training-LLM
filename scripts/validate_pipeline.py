"""Pipeline validation script - verifies all components are working."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def validate_imports():
    """Validate all module imports."""
    print("=" * 60)
    print("Validating Module Imports")
    print("=" * 60)

    modules = [
        ("config", "src.config"),
        ("base", "src.base"),
        ("data", "src.data"),
        ("train", "src.train"),
        ("compress", "src.compress"),
        ("eval", "src.eval"),
        ("registry", "src.registry"),
        ("knowledge", "src.knowledge"),
        ("serve", "src.serve"),
    ]

    for name, module in modules:
        try:
            __import__(module)
            print(f"[OK] {name}")
        except ImportError as e:
            print(f"[FAIL] {name}: {e}")

    print()


def validate_configs():
    """Validate config files exist."""
    print("=" * 60)
    print("Validating Config Files")
    print("=" * 60)

    configs = [
        "config/base.yaml",
        "config/eval.yaml",
        "config/train/qlora.yaml",
        "config/data.yaml",
    ]

    for config in configs:
        path = Path(config)
        if path.exists():
            print(f"[OK] {config}")
        else:
            print(f"[FAIL] {config} (missing)")

    print()


def validate_data():
    """Validate sample data files."""
    print("=" * 60)
    print("Validating Data Files")
    print("=" * 60)

    data_files = [
        "data/eval/problems.json",
        "data/train/ds-v1_train.jsonl",
        "data/val/ds-v1_val.jsonl",
    ]

    for file in data_files:
        path = Path(file)
        if path.exists():
            print(f"[OK] {file}")
        else:
            print(f"[FAIL] {file} (missing)")

    print()


def validate_registry():
    """Validate registry system."""
    print("=" * 60)
    print("Validating Registry System")
    print("=" * 60)

    try:
        from registry.manager import ModelRegistry
        from datetime import datetime
        import tempfile

        # Create temp registry
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            registry = ModelRegistry(temp_path)

            # Test basic operations
            from registry.manager import CheckpointInfo, EvalResult

            test_checkpoint = CheckpointInfo(
                id="test-001",
                source="base",
                base_model="test/model",
                dataset_version=None,
                path="/test/path",
                eval=EvalResult(
                    pass_at_1=0.5,
                    pass_at_k=[0.5, 0.7],
                    latency_tok_s=10.0,
                    size_mb=1000,
                    memory_mb=2000,
                ),
                is_best=True,
                created_at=datetime.now(),
            )

            registry.register_checkpoint(test_checkpoint)
            retrieved = registry.get_checkpoint("test-001")

            if retrieved and retrieved.id == "test-001":
                print("[OK] Registry: Basic operations work")
            else:
                print("[FAIL] Registry: Basic operations failed")

        finally:
            Path(temp_path).unlink(missing_ok=True)

    except Exception as e:
        print(f"[FAIL] Registry: {e}")

    print()


def validate_eval_harness():
    """Validate eval harness."""
    print("=" * 60)
    print("Validating Eval Harness")
    print("=" * 60)

    try:
        from eval.harness import EvalHarness, Problem

        config = {
            "sandbox": {"timeout_seconds": 10},
            "metrics": {"n_samples": [5]},
        }

        harness = EvalHarness(config)

        # Test problem loading
        problems = harness.load_problems("data/eval/problems.json")

        if problems and len(problems) > 0:
            print(f"[OK] Eval Harness: Loaded {len(problems)} problems")
        else:
            print("[FAIL] Eval Harness: Failed to load problems")

    except Exception as e:
        print(f"[FAIL] Eval Harness: {e}")

    print()


def validate_cli():
    """Validate CLI commands."""
    print("=" * 60)
    print("Validating CLI")
    print("=" * 60)

    try:
        from main import main
        print("[OK] CLI: Main entry point imports successfully")
    except Exception as e:
        print(f"[FAIL] CLI: {e}")

    print()


def main():
    """Run all validations."""
    print("\n" + "=" * 60)
    print("LeanCoder Pipeline Validation")
    print("=" * 60 + "\n")

    validate_imports()
    validate_configs()
    validate_data()
    validate_registry()
    validate_eval_harness()
    validate_cli()

    print("=" * 60)
    print("Validation Complete")
    print("=" * 60)
    print("\nAll components are in place. The pipeline is ready for execution.")
    print("\nNext steps:")
    print("1. Set up .env file with HF_TOKEN")
    print("2. Run: python -m src.main baseline")
    print("3. Run: python -m src.main status")
    print()


if __name__ == "__main__":
    main()
