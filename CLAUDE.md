# CLAUDE.md

This file provides guidance to Claude Code (and compatible AI coding agents) when working with this repository.

## Project Summary

**LeanCoder** is a project to clone a strong small open-source LLM and adapt it into a **lightweight, coding-focused Q&A model**. It does NOT pretrain from scratch. Instead it follows the realistic, high-impact pipeline:

**Clone base → Fine-tune (LoRA/QLoRA + SFT, optional DPO/distillation) → Compress (quantization) → Evaluate (executable benchmarks) → Serve.**

The model gets smarter over time through a **gated, versioned data + evaluation loop**: curated coding examples and real failure cases accumulate, every fine-tune is benchmarked, and only measurable improvements are promoted.

## Core Principles

- **Adapt, don't pretrain.** Start from a proven open base model. No from-scratch pretraining.
- **Measure everything.** "Smarter" must be proven by executable coding benchmarks (pass@k), not vibes.
- **Light but good.** Favor quantization + distillation to shrink size while preserving capability.
- **Reproducibility.** Every run logs base model, dataset version, hyperparameters, seed, and eval scores.
- **Promote only improvements.** A new checkpoint replaces the current "best" only if it beats it on the eval suite.
- **Budget-aware.** Target single/consumer-GPU feasibility (QLoRA, gradient checkpointing).

## Architecture Overview

```
src/
  base/            # Base-model download, license check, config
  data/            # Dataset builders, cleaners, dedup, formatting (chat/code templates)
  train/           # SFT, LoRA/QLoRA, DPO, distillation pipelines
  compress/        # Quantization (GGUF/AWQ/GPTQ), optional pruning
  eval/            # Executable benchmarks (HumanEval/MBPP-style), sandbox runner
  serve/           # Inference server (llama.cpp/vLLM/transformers), API
  registry/        # Model/checkpoint registry, versioning, "best" pointer
  knowledge/       # Gated data+eval improvement loop (Second Brain)
  config/          # Env, hyperparams, paths, eval thresholds
artifacts/         # Checkpoints, adapters, quantized models (gitignored)
outputs/           # Timestamped eval reports + training logs
tests/             # Unit tests + eval-harness sanity tests
```

## Key Commands

```bash
# Environment
pip install -e .

# Pipeline stages
python -m src.base.fetch --model <hf-repo-id>       # clone base (license-checked)
python -m src.data.build --recipe coding_sft        # build dataset version
python -m src.train.sft --config configs/qlora.yaml # fine-tune (QLoRA)
python -m src.train.dpo --config configs/dpo.yaml   # optional preference tuning
python -m src.compress.quantize --method awq --bits 4
python -m src.eval.run --suite humaneval,mbpp       # executable benchmarks
python -m src.serve.api --model artifacts/best      # serve current best

# Quality gates
pytest
python -m src.eval.run --quick                      # fast sanity eval
```

## Pipeline Stages (Enforce Order)

1. **Base Clone** — Download a permitted open base; record license + architecture.
2. **Data Build** — Curate/clean/dedup coding Q&A; format with chat/code template; **version it**.
3. **Fine-Tune** — SFT via LoRA/QLoRA (base frozen); optional DPO; optional distillation from a teacher.
4. **Compress** — Quantize (4/8-bit GGUF/AWQ/GPTQ); optional careful pruning.
5. **Evaluate** — Run executable benchmarks in a sandbox; compute pass@k, latency, size.
6. **Promote** — Update registry "best" pointer ONLY if it beats current best on the suite.
7. **Serve** — Expose the best model via an inference API.

## Coding & Training Conventions

- All hyperparameters, dataset recipes, and eval thresholds live in `config/` — never hard-code.
- Every training run writes a manifest: base id, dataset version, hyperparams, seed, hardware, eval scores.
- Datasets are **versioned and append-only by version**; never silently mutate a released dataset.
- Generated code is evaluated in an **isolated sandbox** (no host access).
- Keep adapters separate from base weights for cheap experimentation.

## Evaluation Rules (Mandatory)

- Correctness is measured by **running generated code against tests** (pass@1 / pass@k).
- Track a fixed **held-out eval set**; never train on eval data (prevent leakage).
- Report size (params/disk), latency (tokens/s), and memory alongside accuracy.
- A checkpoint is "better" only if it improves the **primary metric without regressing** guardrail metrics beyond tolerance.

## Guardrails (Mandatory)

- Respect base-model + dataset **licenses**; record provenance for every data source.
- Never execute model-generated code outside the sandbox.
- Never train on the held-out eval set (leakage = invalid results).
- Never promote a checkpoint that fails the eval gate.
- Do not commit large artifacts, weights, API keys, or `.env`.

## Do Not

- Do not attempt from-scratch pretraining.
- Do not overwrite previous eval reports (append-only, timestamped).
- Do not mix dataset versions within a single training run.
- Do not promote on subjective judgment — only on eval-suite results.
