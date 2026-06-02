# Phase 0 Setup — Base Clone + Eval Harness

**Date:** 2026-06-02
**Status:** Approved
**Phase:** 0 — Setup + Base Clone + Eval Harness

---

## Overview

Phase 0 establishes the project foundation and creates a working executable benchmark before any training. This gives immediate confidence the pipeline works and provides a baseline for all future comparisons.

**Goal:** Foundation + a working *executable* benchmark before any training.

**Approach:** Sequential — Track A (Foundation) first, then Track B (Base Model + Eval Harness).

---

## Track A: Foundation Structure

### 1. Repository Structure

```
lean-coder-llm/
├── src/
│   ├── base/           # Model download, license check, config recording
│   ├── data/           # Dataset builders, cleaners, dedup, formatting
│   ├── train/          # SFT, LoRA/QLoRA, DPO, distillation pipelines
│   ├── compress/       # Quantization (GGUF/AWQ/GPTQ), pruning
│   ├── eval/           # Executable benchmarks, sandbox runner
│   ├── serve/          # Inference server, API
│   ├── registry/       # Model/checkpoint registry, versioning, "best" pointer
│   ├── knowledge/      # Gated data+eval improvement loop
│   └── config/         # Env, hyperparams, paths, eval thresholds
├── artifacts/          # Checkpoints, adapters, quantized models (gitignored)
├── outputs/            # Timestamped eval reports + training logs
├── tests/              # Unit tests + eval harness sanity tests
├── config/             # YAML configs for different stages
├── .env.example        # Environment variables template
├── pyproject.toml      # Python package with dependencies
├── CLAUDE.md           # Project instructions (exists)
├── PROJECT-DETAIL.md  # Project spec (exists)
├── PROJECT-DEVELOPMENT-PHASE-TRACKING.md  # Progress tracker (exists)
└── SECOND-KNOWLEDGE-BRAIN.md  # Improvement ledger (exists)
```

### 2. Configuration System

**`.env` (gitignored) - Secrets and local paths:**
```bash
# HuggingFace
HF_TOKEN=your_token_here
HF_HUB_DISABLE_TELEMETRY=1

# Paths
ARTIFACTS_DIR=artifacts
OUTPUTS_DIR=outputs
REGISTRY_PATH=registry/models.json

# GPU/CUDA
CUDA_VISIBLE_DEVICES=0
MAX_GPU_MEM_GB=80

# Eval thresholds
EVAL_PASS_AT_1_THRESHOLD=0.01
EVAL_MIN_SAMPLES=10
```

**`config/base.yaml` - Base model configuration:**
```yaml
model:
  repo_id: "Qwen/Qwen2.5-Coder-7B-Instruct"
  revision: "main"
  expected_license: "apache-2.0"
  trust_remote_code: true

download:
  use_auth_token: true
  save_path: "artifacts/base/Qwen2.5-Coder-7B-Instruct"
```

**`config/eval.yaml` - Evaluation configuration:**
```yaml
sandbox:
  timeout_seconds: 30
  memory_limit_mb: 512
  allow_network: false

metrics:
  n_samples: [10, 50]  # For pass@k calculation
  temperature: 0.2

problems:
  # Path to eval problems (HumanEval/MBPP format)
  path: "data/eval/problems.json"
```

### 3. Core Dependencies (pyproject.toml)

```toml
[project]
name = "lean-coder-llm"
version = "0.0.1"
requires-python = ">=3.10"
dependencies = [
    # Core ML
    "torch>=2.4.0",
    "transformers>=4.43.0",
    "tokenizers>=0.15.0",
    "accelerate>=0.33.0",
    "datasets>=3.0.0",

    # Efficient fine-tuning
    "peft>=0.12.0",
    "bitsandbytes>=0.43.0",

    # Training utilities
    "trl>=0.9.0",

    # Configuration
    "python-dotenv>=1.0.0",
    "pyyaml>=6.0.0",

    # Testing
    "pytest>=8.0.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
```

### 4. Module Stubs

Each module folder gets:
- `__init__.py` with exported interfaces
- Main module file with function signatures
- Type definitions for clarity

---

## Track B: Base Model + Eval Harness

### 1. Base Model Cloner (`src/base/fetch.py`)

**Purpose:** Download a base model from HuggingFace with license verification and record its metadata.

**Interface:**
```python
from dataclasses import dataclass
from pathlib import Path

@dataclass
class ModelInfo:
    """Recorded information about a downloaded model."""
    repo_id: str
    revision: str
    license: str
    architecture: str  # e.g., "Qwen2ForCausalLM"
    context_length: int
    vocab_size: int
    path: Path
    size_mb: float

def download_base_model(config: dict) -> ModelInfo:
    """Download base model from HuggingFace with license verification.

    Args:
        config: Configuration dict from config/base.yaml

    Returns:
        ModelInfo with recorded metadata

    Raises:
        LicenseError: If model license doesn't match expected
        DownloadError: If download fails
    """
    pass
```

**Behavior:**
1. Read config from `config/base.yaml`
2. Authenticate with HF token from `.env`
3. Download model + tokenizer
4. Verify license type matches `expected_license`
5. Extract: architecture, context length, vocab size
6. Save to `artifacts/base/<model-name>/`
7. Return `ModelInfo` for registry

### 2. Model Registry (`src/registry/manager.py`)

**Purpose:** Versioned checkpoint tracking with "best" pointer promotion logic.

**Interface:**
```python
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

@dataclass
class EvalResult:
    """Evaluation metrics for a checkpoint."""
    pass_at_1: float
    pass_at_k: list[float]
    latency_tok_s: float
    size_mb: float
    memory_mb: float

@dataclass
class CheckpointInfo:
    """Registry entry for a checkpoint."""
    id: str
    source: str  # "base", "qlora-sft", "dpo", "quantized"
    base_model: str
    dataset_version: Optional[str]
    path: str
    eval: Optional[EvalResult]
    is_best: bool
    created_at: datetime
    manifest: dict = field(default_factory=dict)

class ModelRegistry:
    """Manages versioned checkpoint registry."""

    def __init__(self, registry_path: str):
        """Load or create registry file."""
        pass

    def register_checkpoint(self, info: CheckpointInfo) -> str:
        """Add a new checkpoint to registry.

        Returns:
            Checkpoint ID
        """
        pass

    def get_best(self) -> Optional[CheckpointInfo]:
        """Get the current best checkpoint."""
        pass

    def promote_if_better(self, new_id: str, primary_metric: str = "pass_at_1") -> bool:
        """Promote new checkpoint to "best" if it beats current best.

        Args:
            new_id: Checkpoint ID to evaluate
            primary_metric: Metric to compare (default: pass_at_1)

        Returns:
            True if promoted, False otherwise
        """
        pass
```

**Registry Storage:** `registry/models.json`
```json
{
  "checkpoints": [
    {
      "id": "base-20260602-100000",
      "source": "base",
      "base_model": "Qwen/Qwen2.5-Coder-7B-Instruct",
      "dataset_version": null,
      "path": "artifacts/base/Qwen2.5-Coder-7B-Instruct",
      "eval": {
        "pass_at_1": 0.35,
        "pass_at_k": [0.35, 0.52, 0.63],
        "latency_tok_s": 45.2,
        "size_mb": 14500
      },
      "is_best": true,
      "created_at": "2026-06-02T10:00:00Z"
    }
  ],
  "best_pointer": "base-20260602-100000"
}
```

### 3. Eval Harness (`src/eval/harness.py`)

**Purpose:** Run executable coding benchmarks in a sandboxed environment.

**Interface:**
```python
from dataclasses import dataclass

@dataclass
class Problem:
    """A coding problem with test cases."""
    id: str
    description: str
    starter_code: str
    tests: list[str]  # Test functions as strings

@dataclass
class EvalResult:
    """Results from running eval on a model."""
    suite: str
    pass_at_1: float
    pass_at_k: list[float]
    latency_tok_s: float
    model_size_mb: float
    memory_mb: float
    problems_total: int
    details: list[dict]

class EvalHarness:
    """Executable code evaluation harness."""

    def __init__(self, config: dict):
        """Initialize harness with sandbox config."""
        pass

    def load_problems(self, path: str) -> list[Problem]:
        """Load problems from JSON file."""
        pass

    def run_eval(self, model, problems: list[Problem], n_samples: int) -> EvalResult:
        """Run evaluation on model against problems.

        Args:
            model: Loaded model for generation
            problems: List of coding problems
            n_samples: Number of samples per problem for pass@k

        Returns:
            EvalResult with metrics
        """
        pass

    def _generate_solution(self, problem: Problem, n: int) -> list[str]:
        """Generate n code solutions for a problem."""
        pass

    def _execute_sandboxed(self, code: str, tests: list[str]) -> bool:
        """Execute code against tests in isolated sandbox.

        Returns:
            True if all tests pass
        """
        pass
```

**Sandbox Behavior:**
- Isolated process with no host access
- Timeout limit (configurable, default 30s)
- Memory limit (configurable, default 512MB)
- No network access
- Captures stdout/stderr for debugging

### 4. Eval Reporter (`src/eval/reporter.py`)

**Purpose:** Write timestamped, comparable eval reports.

**Interface:**
```python
from datetime import datetime
from pathlib import Path

class EvalReporter:
    """Writes timestamped eval reports."""

    def __init__(self, outputs_dir: str):
        """Initialize reporter with outputs directory."""
        pass

    def write_report(
        self,
        result: EvalResult,
        checkpoint_id: str,
        comparison: dict
    ) -> Path:
        """Write eval report to outputs/.

        Args:
            result: EvalResult from harness
            checkpoint_id: ID of evaluated checkpoint
            comparison: Dict with vs_best and vs_base deltas

        Returns:
            Path to written report
        """
        pass

    def _format_comparison(self, comparison: dict) -> str:
        """Format comparison deltas as markdown table."""
        pass
```

**Report Format:** `outputs/eval/2026-06-02_10-30-00_eval.md`
```markdown
# Evaluation Report — 2026-06-02 10:30:00

**Checkpoint:** base-20260602-100000
**Source:** Base Model
**Suite:** HumanEval + MBPP

## Results

| Metric | Value |
|--------|-------|
| Pass@1 | 35.0% |
| Pass@10 | 63.0% |
| Latency | 45.2 tok/s |
| Size | 14500 MB |
| Memory | 28000 MB |

## Comparison

| vs. | Pass@1 Delta | Pass@10 Delta | Size Delta |
|-----|--------------|---------------|------------|
| Best | — | — | — |
| Base | — | — | — |

## Details

- Total problems: 200
- Samples per problem: 50
- Timeout: 30s per problem
```

### 5. Baseline Runner (`src/eval/baseline.py`)

**Purpose:** Orchestrate the complete baseline evaluation flow.

**Interface:**
```python
def run_baseline(
    model_config: dict,
    eval_config: dict,
    registry: ModelRegistry
) -> CheckpointInfo:
    """Run complete baseline: download → eval → record → promote.

    Args:
        model_config: Base model config
        eval_config: Evaluation config
        registry: Model registry instance

    Returns:
        Registered baseline checkpoint info
    """
    pass
```

**Flow:**
1. Download base model (via `base/fetch.py`)
2. Create `CheckpointInfo` for baseline
3. Run eval harness (via `eval/harness.py`)
4. Write report (via `eval/reporter.py`)
5. Register checkpoint
6. Promote to "best" (it's the first, so auto-promote)

---

## Data Model

### CheckpointInfo (Registry Entry)
```json
{
  "id": "checkpoint-YYYYMMDD-HHMMSS",
  "source": "base" | "qlora-sft" | "dpo" | "quantized",
  "base_model": "Qwen/Qwen2.5-Coder-7B-Instruct",
  "dataset_version": "ds-v1" | null,
  "path": "artifacts/...",
  "eval": {
    "pass_at_1": 0.35,
    "pass_at_k": [0.35, 0.52, 0.63],
    "latency_tok_s": 45.2,
    "size_mb": 14500
  },
  "is_best": true,
  "created_at": "2026-06-02T10:30:00Z",
  "manifest": {}
}
```

### EvalResult
```json
{
  "suite": "human_eval",
  "pass_at_1": 0.35,
  "pass_at_k": [0.35, 0.52, 0.63],
  "latency_tok_s": 45.2,
  "model_size_mb": 14500,
  "memory_mb": 28000,
  "problems_total": 200,
  "details": [
    {"problem_id": "0", "passed": true, "samples": 50, "c": 35}
  ]
}
```

---

## Definition of Done

### Track A: Foundation
- [ ] All folders exist with `__init__.py`
- [ ] Module stubs define interfaces for all pipeline stages
- [ ] `pyproject.toml` with all dependencies
- [ ] `.env.example` template with all variables
- [ ] Config YAML files created (`base.yaml`, `eval.yaml`)
- [ ] `pytest` runs with 0 errors (test the stubs)
- [ ] `.gitignore` excludes `artifacts/`, `outputs/`, `.env`

### Track B: Base Model + Eval
- [ ] Qwen2.5-Coder-7B-Instruct downloaded to `artifacts/base/`
- [ ] Registry initialized at `registry/models.json`
- [ ] Baseline checkpoint registered and promoted to "best"
- [ ] Eval harness generates and executes code in sandbox
- [ ] Baseline report written to `outputs/eval/`
- [ ] Unit tests for registry, harness, reporter

---

## Guardrails

- **No secrets in repo:** `.env` is gitignored; `.env.example` is template only
- **License verification:** Base model download fails if license doesn't match expected
- **Sandbox isolation:** Eval harness runs code in isolated process with limits
- **No eval leakage:** Holdout set separate from training (enforced in Phase 1)
- **Promotion by metrics only:** Registry promotes only on measurable improvement

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| HF token not available | Graceful error with setup instructions |
| Download size too large | Support smaller model variants (0.5B, 1.5B) |
| Sandbox escape | Use multiprocessing with resource limits |
| GPU memory insufficient | Support CPU-only for baseline eval |

---

## Next Steps

After Phase 0 completion:
- **Phase 1:** Versioned data build (clean, dedup, leakage-check)
- **Phase 2:** QLoRA SFT fine-tuning pipeline
- **Phase 3:** Compression via quantization
- **Phase 4:** Promotion registry (refined from Phase 0)
- **Phase 4.5:** Knowledge Core (gated improvement loop)
- **Phase 5:** DPO / distillation
- **Phase 6:** Serving + API
- **Phase 7:** v1.0 release
