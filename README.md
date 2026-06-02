# <img src="media/opencli.png" alt="LeanCoder Logo" width="200"/>

# LeanCoder: Lean, Adaptable LLM Fine-Tuning Pipeline

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.4+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black.svg)](https://github.com/psf/black)

**LeanCoder** is a lightweight, production-ready pipeline for fine-tuning small open-source LLMs into capable coding assistants. Built on proven techniques—no massive pretraining required.

## 🎯 Mission

Democratize LLM fine-tuning by providing a **complete, reproducible pipeline** that:
- ✅ Works on consumer hardware (single GPU)
- ✅ Measures everything via executable benchmarks
- ✅ Improves monotonically through gated data+eval loops
- ✅ Ships production-ready inference

## 🌟 Key Features

| Feature | Description |
|---------|-------------|
| **QLoRA Fine-Tuning** | Train on 4-bit quantized base—7B models on 24GB VRAM |
| **Docker Sandboxed Eval** | Safe code execution with resource limits |
| **Comprehensive Benchmarks** | HumanEval (164), MBPP (974), LeetCode-style problems |
| **Resumable Training** | Save/restore from any checkpoint—no lost progress |
| **Multi-GPU Support** | FSDP and DeepSpeed for distributed training |
| **Real-time Dashboard** | Web UI for monitoring training & metrics |
| **Auto Hyperparameter Search** | Optuna-powered optimization |
| **Model Ensemble** | Combine top checkpoints via weighted voting |
| **Knowledge Core** | Gated improvement loop for continuous gains |
| **Production API** | FastAPI inference server ready for deployment |

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/dungnotnull/lean-coder-self-training-LLM.git
cd lean-coder-self-training-LLM

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -e .
```

### Configuration

```bash
cp .env.example .env
# Edit .env with your HF_TOKEN and other settings
```

### Run Pipeline

```bash
# View status
python -m src.main status

# Run baseline evaluation
python -m src.main baseline

# Build dataset version
python -m src.main data-build \
  --version ds-v1 \
  --input data/raw/coding_qa.jsonl

# Fine-tune model
python -m src.main train \
  --model artifacts/base/Qwen2.5-Coder-7B-Instruct \
  --train-data data/train/ds-v1_train.jsonl \
  --val-data data/val/ds-v1_val.jsonl \
  --dataset-version ds-v1

# Quantize model
python -m src.main quantize \
  --model artifacts/base/Qwen2.5-Coder-7B-Instruct \
  --method gguf --bits 4

# Promote if better
python -m src.main promote \
  --checkpoint-id qlora-20260602-120000

# Start inference server
python -m src.main serve --host 0.0.0.0 --port 8000
```

## 📁 Architecture

```
lean-coder-llm/
├── src/
│   ├── base/              # Model download, license verification
│   ├── data/              # Dataset build, clean, dedup, augment
│   ├── train/             # SFT, QLoRA, DPO, distributed training
│   ├── compress/          # Quantization, pruning
│   ├── eval/              # Benchmarks, sandbox, A/B testing
│   ├── serve/             # Inference server, dashboard, ensemble
│   ├── registry/          # Checkpoint versioning, promotion logic
│   ├── knowledge/         # Gated improvement loop
│   ├── plugins/           # Extensible plugin system
│   ├── models/            # Model abstraction layer
│   ├── pipeline/          # Async pipeline orchestration
│   └── config/            # Configuration management
├── config/                # YAML configs
├── notebooks/             # Interactive tutorials
├── tests/                 # Unit tests
├── docs/                  # Design specs
└── scripts/               # Utility scripts
```

## 🧪 Evaluation

LeanCoder evaluates via **executable code**—no vibes, only measured results.

### Benchmarks

- **HumanEval**: 164 hand-written Python problems
- **MBPP**: 974 Mostly Basic Python Problems
- **LeetCode-style**: Algorithm challenges
- **Custom**: Domain-specific problems

### Metrics

- **Pass@1**: Primary metric—first attempt correctness
- **Pass@10**: Resilience—correctness in 10 attempts
- **Latency**: Tokens per second
- **Size**: Model footprint

### Sandbox

All code executes in **Docker containers** with:
- ⏱️ Execution timeout
- 💾 Memory limits
- 🚫 No network access
- 🧹 Auto-cleanup

## 🔄 Pipeline Stages

### Phase 0: Setup + Baseline
Download base model → Establish baseline metrics

### Phase 1: Data Build
Collect → Clean → Dedup → Format → Version dataset

### Phase 2: QLoRA Fine-Tuning
Load 4-bit base → Train LoRA adapters → Save checkpoint

### Phase 3: Compression
Quantize (GGUF/AWQ/GPTQ) → Re-evaluate → Document trade-offs

### Phase 4: Promotion Registry
Compare vs. best → Promote only if better → Update pointer

### Phase 4.5: Knowledge Core
Capture failures → Quality gate → New dataset → Retrain → Measure

### Phase 5: DPO/Distillation
Preference optimization or teacher-student distillation

### Phase 6: Serving
Load best model → Start FastAPI server → Query via API

### Phase 7: Release
Documentation → Reproducibility checks → Tag v1.0

## 🎓 Tutorials

Interactive Jupyter notebooks in `notebooks/`:

1. **01_baseline.ipynb** - Run baseline evaluation
2. **02_finetune.ipynb** - QLoRA fine-tuning
3. **03_eval.ipynb** - Comprehensive evaluation

```bash
jupyter notebook notebooks/01_baseline.ipynb
```

## 🔧 Advanced Features

### Hyperparameter Search

```python
from train.optuna import run_hyperparameter_search

best_params = run_hyperparameter_search(
    base_model_path="artifacts/base/Qwen2.5-Coder-7B-Instruct",
    train_data="data/train/ds-v1_train.jsonl",
    val_data="data/val/ds-v1_val.jsonl",
    n_trials=50,
)
```

### Model Ensemble

```python
from serve.ensemble import ModelEnsemble

ensemble = ModelEnsemble.create_from_registry(
    registry_path="registry/models.json",
    top_n=3,
    weight_strategy="score_based",
)

response = ensemble.predict("Write a function to add two numbers")
```

### A/B Testing

```python
from eval.ab_testing import ABTestFramework

framework = ABTestFramework("outputs/ab_tests.json")

test = framework.create_test(
    test_id="test_001",
    name="QLoRA vs DPO",
    variant_a_path="artifacts/qlora/model",
    variant_b_path="artifacts/dpo/model",
    test_cases=test_cases,
)

results = framework.run_test("test_001")
```

### Docker Dashboard

```bash
# Start dashboard
python -m src.serve.dashboard.app

# Access at http://localhost:8001
```

Features:
- 📊 Real-time training curves
- 📈 Registry history
- 🧠 Knowledge brain timeline
- 📉 Loss & metrics tracking

## 🧩 Plugin System

Extend LeanCoder with plugins:

```python
from src.plugins import Plugin, PluginManager

class MyPlugin(Plugin):
    @property
    def name(self):
        return "my_plugin"

    @property
    def version(self):
        return "1.0.0"

    def on_eval_end(self, context, results):
        print(f"Eval complete: {results['pass_at_1']}")

manager = PluginManager()
manager.register_plugin(MyPlugin())
```

Built-in plugins:
- **Weights & Biases** - Experiment tracking
- **Slack Notifier** - Notifications
- **Metrics Exporter** - Export to JSON/CSV

## 📊 Monitoring

### Metrics Collection

```python
from serve.dashboard.metrics import TrainingMetrics

metrics = TrainingMetrics("outputs/metrics.json")

# During training
metrics.record_loss(loss=2.5, step=100)
metrics.record_eval_score(score=0.42, step=100)
```

### Progress Tracking

```python
from train.resumable import ProgressTracker

tracker = ProgressTracker(total_steps=1000)
tracker.start()

for step in range(1000):
    tracker.update(step, {"loss": current_loss})
```

## 🌐 Web Dashboard

```bash
python -m src.serve.dashboard.app
```

- 📈 Real-time metrics
- 📋 Registry view
- 🧠 Knowledge brain timeline
- 📊 Training curves

## 🚦 Inference API

### Start Server

```bash
python -m src.main serve --host 0.0.0.0 --port 8000
```

### API Endpoints

```bash
# Generate completion
curl -X POST http://localhost:8000/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write a function to add two numbers", "max_tokens": 512}'

# Health check
curl http://localhost:8000/health
```

## 🧪 Testing

### Run Tests

```bash
# All tests
pytest

# Specific module
pytest tests/test_registry.py

# With coverage
pytest --cov=src --cov-report=html
```

### Validation

```bash
python scripts/validate_pipeline.py
```

## 📖 Documentation

- **[Phase 0 Design Spec](docs/superpowers/specs/2026-06-02-phase0-setup-design.md)** - System design
- **[PROJECT DETAIL](PROJECT-DETAIL.md)** - Detailed project documentation
- **[PROJECT TRACKING](PROJECT-DEVELOPMENT-PHASE-TRACKING.md)** - Development progress
- **[KNOWLEDGE BRAIN](SECOND-KNOWLEDGE-BRAIN.md)** - Improvement ledger

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/lean-coder-self-training-LLM.git
cd lean-coder-self-training-LLM

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
black src/
isort src/
```

## 📜 License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

Compatible with base model licenses (Qwen2.5-Coder: Apache 2.0).

## 🙏 Acknowledgments

- **Qwen Team** - Qwen2.5-Coder base model
- **HuggingFace** - Transformers, PEFT, datasets
- **Optuna** - Hyperparameter optimization
- **DeepSpeed** - Distributed training
- **FastAPI** - Inference server framework

## 📬 Citation

If you use LeanCoder in your research:

```bibtex
@software{leancoder2024,
  title={LeanCoder: Lean LLM Fine-Tuning Pipeline},
  author={Dung},
  year={2024},
  url={https://github.com/dungnotnull/lean-coder-self-training-LLM},
}
```

## 🗺️ Roadmap

- [ ] Multi-GPU training optimizations
- [ ] Vision-language models support
- [ ] Streaming inference
- [ ] Model merging strategies
- [ ] Advanced data augmentation
- [ ] Integration with more base models

## 📞 Support

- 📖 [Documentation](docs/)
- 🐛 [Issues](https://github.com/dungnotnull/lean-coder-self-training-LLM/issues)
- 💬 [Discussions](https://github.com/dungnotnull/lean-coder-self-training-LLM/discussions)

---

**Built with ❤️ for the open-source community**

Made with [Claude Code](https://claude.com/claude-code)
