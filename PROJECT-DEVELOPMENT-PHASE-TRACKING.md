# PROJECT-DEVELOPMENT-PHASE-TRACKING.md — LeanCoder

> **Purpose:** Single source of truth for development progress. Update at the end of every work session. Each task has a checkbox; each phase has an explicit **Definition of Done (DoD)**.

---

## How to Use This File

- Mark tasks: `[ ]` not started · `[~]` in progress · `[x]` done · `[!]` blocked.
- Never delete completed items — keep history.
- Update **Last Updated** and the dashboard on every edit.
- Close a phase only when **all DoD criteria pass**.

**Last Updated:** 2026-05-31
**Current Phase:** Phase 0 — Setup
**Overall Progress:** 0%

---

## Status Legend

| Symbol | Meaning |
|--------|---------|
| `[ ]` | Not started |
| `[~]` | In progress |
| `[x]` | Completed |
| `[!]` | Blocked (note reason) |

---

## Phase Summary Dashboard

| **Phase** | **Goal** | **Status** | **Target** |
|-----------|----------|------------|------------|
| Phase 0 | Setup + base clone + eval harness | `[ ]` | Week 1–2 |
| Phase 1 | Versioned data build | `[ ]` | Week 3 |
| Phase 2 | QLoRA SFT fine-tuning | `[ ]` | Week 4–5 |
| Phase 3 | Compression (quantization) | `[ ]` | Week 6 |
| Phase 4 | Promotion registry + comparisons | `[ ]` | Week 7 |
| Phase 4.5 | Knowledge Core (gated improvement loop) | `[ ]` | Week 8 |
| Phase 5 | DPO / distillation (capability gains) | `[ ]` | Week 9–10 |
| Phase 6 | Serving + API | `[ ]` | Week 11 |
| Phase 7 | v1.0 release | `[ ]` | Week 12 |

---

## Phase 0 — Setup + Base Clone + Eval Harness

**Goal:** Foundation + a working *executable* benchmark before any training.

- [ ] Initialize repo, license, `.gitignore` (ignore `artifacts/`)
- [ ] PyTorch + Transformers + PEFT + TRL + bitsandbytes installed
- [ ] Folder structure (`base`, `data`, `train`, `compress`, `eval`, `serve`, `registry`, `knowledge`, `config`)
- [ ] `.env.example` (HF token, paths, GPU config, eval thresholds)
- [ ] **Clone base model** (license-checked) + record architecture/tokenizer
- [ ] **Sandboxed eval harness** (HumanEval/MBPP-style) that runs generated code against tests
- [ ] Record **base baseline** pass@1/pass@k + size/latency
- [ ] CI: lint + unit tests + quick eval sanity

**Definition of Done**
- Base model loads and generates.
- Eval harness runs in sandbox and produces a baseline report.

---

## Phase 1 — Versioned Data Build

**Goal:** Clean, deduped, leakage-free, versioned coding dataset.

- [ ] Collect permissively-licensed code + instruction/Q&A sources (record provenance)
- [ ] Clean: remove secrets/PII, filter low quality
- [ ] Dedup (exact + near-duplicate)
- [ ] Chat/code template formatting
- [ ] Train/val/holdout split with **automated leakage check** vs. eval set
- [ ] Tag dataset version `ds-v1` + stats report

**Definition of Done**
- `ds-v1` exists with provenance, dedup stats, and verified no eval leakage.

---

## Phase 2 — QLoRA SFT Fine-Tuning

**Goal:** Adapt the base on coding data, single-GPU feasible.

- [ ] QLoRA config (4-bit base, adapter ranks, LR, scheduler)
- [ ] SFT training loop with gradient checkpointing
- [ ] Run manifest logging (base, ds version, hyperparams, seed, hardware)
- [ ] Evaluate fine-tuned checkpoint vs. base
- [ ] Write `outputs/..._train.md` + `outputs/..._eval.md`

**Definition of Done**
- Fine-tuned checkpoint **beats base** on pass@1 (or documented why not), with full manifest.

---

## Phase 3 — Compression (Quantization)

**Goal:** Light but still good.

- [ ] Quantize to 4/8-bit (GGUF and/or AWQ/GPTQ)
- [ ] Re-evaluate quantized model (quality + size + latency)
- [ ] Optional careful pruning + re-eval
- [ ] Write `outputs/..._compress.md` (before/after)

**Definition of Done**
- Quantized model documented with quality-vs-size/latency trade-off; minimal acceptable quality loss.

---

## Phase 4 — Promotion Registry + Comparisons

**Goal:** Only better models become "best."

- [ ] Model registry with `best_pointer` + checkpoint history
- [ ] Promotion rule: beat current best on primary metric, no guardrail regression
- [ ] Automated comparison report (new vs. best vs. base)

**Definition of Done**
- Promotion is automatic, rule-based, and logged; worse checkpoints are rejected.

---

## Phase 4.5 — Knowledge Core (Gated Improvement Loop)

**Goal:** Measurable, reproducible self-improvement over time.

- [ ] Capture **real failure cases** (wrong/incomplete answers) into a candidate pool
- [ ] Quality gate: relevance, correctness of target solution, license, dedup, **leakage check**
- [ ] Promote accepted examples into a new **dataset version**
- [ ] Re-fine-tune → evaluate → promote only if benchmark improves
- [ ] Append a versioned entry to `SECOND-KNOWLEDGE-BRAIN.md`
- [ ] Guardrail test: held-out eval never contaminated by new data

**Definition of Done**
- A full loop iteration produces a new dataset version + checkpoint that is promoted *only* on measured improvement, logged to the brain.

---

## Phase 5 — DPO / Distillation (Capability Gains)

**Goal:** Push quality at small size.

- [ ] Build preference pairs (chosen/rejected) for DPO
- [ ] DPO training + eval vs. SFT best
- [ ] Optional: distillation from a larger teacher → student + eval
- [ ] Promote if improved

**Definition of Done**
- DPO and/or distillation checkpoint evaluated; promoted only if it beats prior best.

---

## Phase 6 — Serving + API

**Goal:** Make the best model usable.

- [ ] Inference server (llama.cpp / vLLM / transformers)
- [ ] Simple chat/completion API for coding Q&A
- [ ] Loads current registry "best" model
- [ ] Latency/throughput check

**Definition of Done**
- API serves the best model; coding Q&A works end to end within latency target.

---

## Phase 7 — v1.0 Release

**Goal:** Polished, reproducible, documented.

- [ ] Quickstart + full pipeline docs + eval methodology
- [ ] Reproducibility check (rerun a training+eval from manifest)
- [ ] End-to-end demo (clone → FT → quantize → eval → serve)
- [ ] Tag `v1.0.0` + changelog
- [ ] Sync `PROJECT-detail.md` and this tracker to final state

**Definition of Done**
- A new user runs the full pipeline from the README and reproduces reported scores.

---

## Open Issues / Blockers Log

| **Date** | **Issue** | **Phase** | **Status** | **Notes** |
|----------|-----------|-----------|------------|-----------|
| — | — | — | — | — |

---

## Decision Log

| **Date** | **Decision** | **Rationale** |
|----------|--------------|---------------|
| 2026-05-31 | Adapt (clone + fine-tune), not pretrain | Pretraining infeasible solo |
| 2026-05-31 | QLoRA as core FT method | Single-GPU feasibility |
| 2026-05-31 | Executable benchmarks as truth | "Smarter" must be measured |
| 2026-05-31 | Promote only on eval improvement | Prevent regressions |
| 2026-05-31 | Self-improvement via gated data+eval loop | Reproducible, monotonic gains |

---

## Session Notes

- **2026-05-31:** Project initialized. Four docs created. Scope locked to clone→FT→compress→eval→serve with a gated improvement loop. Next: Phase 0 setup + eval harness first.
