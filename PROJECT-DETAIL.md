# PROJECT-detail.md — LeanCoder

## 1. Vision

LeanCoder turns a strong **small open-source LLM** into a **lightweight, coding-focused Q&A assistant** that is cheap to run yet genuinely capable. It is built on the realistic path that individuals and small teams can actually execute: **clone → fine-tune → compress → evaluate → serve**, with continuous, measurable improvement over time.

Its signature is **honesty about capability**: every "improvement" is proven by *executable benchmarks*, and the model evolves through a *gated, versioned data + eval loop* — never opaque claims.

---

## 2. What "Train" Realistically Means (The Core Honesty)

Pretraining from scratch is infeasible (millions of dollars, huge clusters). LeanCoder therefore **adapts** a proven base. This is not a limitation — it's the smart path.

| **Approach** | **Cost** | **Feasible solo?** | **Used here?** |
|--------------|----------|--------------------|----------------|
| Pretrain from scratch | $$$$$ | ❌ No | ❌ No |
| Full fine-tune | $$$ | ⚠️ Hard | Optional, rare |
| **LoRA / QLoRA fine-tune** | $ | ✅ Yes | ✅ Core |
| **Distillation from teacher** | $$ | ✅ Yes | ✅ Optional |
| **Quantization** | ~free | ✅ Yes | ✅ Core |

---

## 3. Goals & Non-Goals

### Goals
- Clone a permitted small open base model (coding-capable).
- Fine-tune (SFT + LoRA/QLoRA, optional DPO/distillation) for coding Q&A.
- Compress to run on modest hardware (4/8-bit quantization).
- Prove capability via **executable coding benchmarks**.
- Improve continuously through a gated data + eval loop.
- Serve via a simple inference API.

### Non-Goals
- No from-scratch pretraining.
- No claims of frontier-model parity.
- No deployment of unevaluated checkpoints.

---

## 4. Target Users & Value

| **User** | **Value** |
|----------|-----------|
| Solo dev / student | A private, local, fast coding helper that runs on a laptop/GPU |
| Researcher / learner | A transparent, reproducible LLM-adaptation pipeline to study |
| Small team | A cheap, self-hosted coding Q&A model without API costs |

---

## 5. End-to-End Pipeline

### Step 1 — Base Clone
- Choose a small coding-capable open base (e.g., Qwen-Coder, DeepSeek-Coder, StarCoder2, small Llama-class).
- Verify license permits fine-tuning + redistribution of derivatives.
- Record architecture, tokenizer, context length, base eval baseline.

### Step 2 — Data Build (Versioned)
- Sources: permissively-licensed code + instruction datasets, curated Q&A, and **real failure cases** captured over time.
- Clean: dedup (exact + near-dup), remove secrets/PII, filter low quality, license-check.
- Format: consistent chat/code template; train/val split with **no eval leakage**.
- Tag a **dataset version** (e.g., `ds-v3`).

### Step 3 — Fine-Tune
- **SFT** with **QLoRA** (4-bit base frozen, train small adapters) for single-GPU feasibility.
- Optional **DPO** for preference/quality alignment.
- Optional **distillation**: a larger teacher model supervises the student to boost capability at small size.
- Log full manifest (base, dataset version, hyperparams, seed, hardware).

### Step 4 — Compress
- Quantize to 4/8-bit (GGUF for llama.cpp, AWQ/GPTQ for GPU serving).
- Optional careful pruning; re-evaluate after any compression.
- Goal: maximize **quality-per-megabyte** and **tokens/sec**.

### Step 5 — Evaluate (Mandatory Gate)
- Run executable benchmarks (HumanEval/MBPP-style) in a **sandbox**: generate code → run tests → compute **pass@1 / pass@k**.
- Measure size, latency (tokens/s), memory.
- Compare against current "best" and the base baseline.

### Step 6 — Promote
- Update registry **"best" pointer** only if the new checkpoint **beats current best** on the primary metric without unacceptable regressions.

### Step 7 — Serve
- Expose best model via inference server (llama.cpp / vLLM / transformers) + simple API.

---

## 6. Evaluation Framework (The Truth Layer)

For a coding model, correctness must be *executed*, not guessed.

- **Primary metric:** pass@1 / pass@k on held-out coding tasks (code is run against unit tests in a sandbox).
- **Secondary:** instruction-following quality, response latency, model size, memory footprint.
- **Anti-leakage:** held-out eval set is *never* in training data; checked automatically.
- **Reporting:** every run produces a timestamped report with scores + manifest, enabling apples-to-apples comparison across checkpoints.

$$pass@k = \mathbb{E}_{\text{problems}}\left[1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}\right]$$

where $$n$$ is samples generated, $$c$$ is the number that pass tests.

---

## 6.5 Evolving Knowledge Core (Self-Improvement Engine)

LeanCoder gets better over time through a **gated, versioned data + eval loop** — not vague "auto-learning."

```
Capture failures + new examples → Quality gate → Dedup/leakage check
  → Version dataset → Fine-tune → Evaluate (executable)
  → Promote ONLY if benchmark improves → Log to SECOND-KNOWLEDGE-BRAIN.md
```

| **Mechanism** | **LeanCoder (correct)** | **Naive (avoided)** |
|---------------|-------------------------|----------------------|
| What improves | Versioned dataset + measured checkpoints | Unverified weight drift |
| Promotion rule | Must beat eval suite | "Feels better" |
| Reproducibility | Full manifest per run | Lost |
| Leakage control | Held-out set enforced | Contaminated |

This is the system's moat: **measurable, reproducible, monotonic improvement.**

---

## 7. Tech Stack (Suggested)

| **Layer** | **Choice** | **Why** |
|-----------|-----------|---------|
| Framework | PyTorch + Hugging Face Transformers | Standard, rich ecosystem |
| Efficient FT | PEFT (LoRA/QLoRA), bitsandbytes | Single-GPU feasibility |
| Preference tuning | TRL (DPO/SFT trainers) | Mature, documented |
| Quantization | llama.cpp (GGUF), AutoAWQ/GPTQ | Light, fast inference |
| Serving | llama.cpp / vLLM / transformers | Flexible deployment |
| Eval | HumanEval/MBPP harness + sandbox | Executable correctness |
| Tracking | Weights & Biases / local logs | Reproducible experiments |

---

## 8. Data Model (Core Entities)

```jsonc
DatasetVersion {
  id,                 // ds-vN
  sources[],          // with license + provenance
  size, dedup_stats,
  split: { train, val, holdout },
  created_at
}

TrainingRun {
  id, base_model, dataset_version,
  method,             // sft | qlora | dpo | distill
  hyperparams{}, seed, hardware,
  adapter_path, created_at
}

EvalReport {
  run_id, suite, pass_at_1, pass_at_k,
  latency_tok_s, model_size_mb, memory_mb,
  vs_best, vs_base, timestamp
}

ModelRegistry {
  best_pointer, checkpoints[]   // each with EvalReport
}
```

---

## 9. Outputs

Timestamped, append-only, in `outputs/`:
- `*_train.md` — run manifest + training log summary.
- `*_eval.md` — benchmark scores + size/latency + comparison to best/base.
- `*_compress.md` — quantization method + before/after quality.

---

## 10. Risks & Mitigations

| **Risk** | **Mitigation** |
|----------|----------------|
| Expecting frontier quality | Honest scope: lean, coding-focused, measured |
| Eval data leakage | Automated held-out check; never train on eval |
| Quantization quality loss | Re-evaluate after compression; choose best method |
| Overfitting to benchmarks | Diverse held-out tasks; rotate problems |
| License violations | Provenance + license check per data source/model |
| Unsafe code execution | Sandboxed eval only; no host access |
| Promoting a worse model | Hard eval gate before "best" promotion |

---

## 11. Roadmap

1. **MVP** — Clone base + sandboxed eval harness + baseline scores.
2. **v0.2** — Versioned data build (clean/dedup/leakage-checked).
3. **v0.3** — QLoRA SFT pipeline + manifest logging.
4. **v0.4** — Quantization + before/after eval.
5. **v0.5** — Promotion registry ("best" pointer + comparisons).
6. **v0.6** — DPO and/or distillation for capability gains.
7. **v0.6b** — Knowledge Core: gated data+eval improvement loop + `SECOND-KNOWLEDGE-BRAIN.md`.
8. **v0.7** — Inference server + simple API.
9. **v1.0** — Polished pipeline, docs, reproducibility, release.
