# SECOND-KNOWLEDGE-BRAIN.md — LeanCoder Evolving Knowledge Core

> **What this is:** LeanCoder's persistent, versioned record of how the model improves over time. Unlike a RAG knowledge base, here "knowledge" = **curated training data + measured checkpoints**. The model gets smarter because its *dataset and evaluated checkpoints* improve — each step proven by executable benchmarks.
>
> **Golden rule:** Improvement is *measured and promoted*, never assumed. A new checkpoint enters the lineage only if it **beats the current best** on the eval suite, with no eval leakage.

---

## How Improvement Works (Mechanism)

```
Capture failures + new examples → Quality gate (license, correctness, dedup, leakage)
  → New dataset version → Fine-tune → Executable eval (pass@k)
  → Promote ONLY if it beats best → Record entry here (versioned)
```

- **Improvement = better data + better checkpoints**, both versioned.
- **The eval suite is the judge.** No subjective "feels smarter."
- **Held-out eval is sacred.** New data is leakage-checked before training.
- **Full reproducibility.** Every entry links to a dataset version + run manifest + eval report.

---

## Quality Gate (Before Any Example Enters Training)

A candidate example is accepted only if it passes:

- [ ] **License/provenance** — source permits training use; provenance recorded.
- [ ] **Correctness** — the target solution is verified (runs, passes tests) for code examples.
- [ ] **Relevance** — coding Q&A within scope.
- [ ] **Deduplication** — not already in the dataset (exact + near-dup).
- [ ] **Leakage check** — does NOT overlap the held-out eval set.
- [ ] **Quality** — clear prompt, clean solution, no secrets/PII.

Failures go to **Quarantine**, not the dataset.

---

## Knowledge Entry Schema

```jsonc
BrainEntry {
  id,                     // KB-YYYYMMDD-####
  dataset_version,        // ds-vN added/changed
  added_examples,         // count + brief description
  training_run,           // run manifest id
  method,                 // sft | qlora | dpo | distill
  eval { pass_at_1, pass_at_k, latency_tok_s, size_mb },
  vs_previous_best,       // delta on primary metric
  decision,               // promoted | rejected
  brain_version,          // bumped on each promotion
  created_at
}
```

---

## Improvement Ledger (Append-Only, Newest First)

> Every accepted improvement is logged here. Rejected attempts are also recorded (transparency). **Never edit past entries** — append a new one.

### 2026-05-31 — Brain initialized
- **KB-20260531-0001** · *Bootstrap* · brain_version: v1.0
  - Summary: Knowledge core established. Base baseline pending first eval. No checkpoints promoted yet.
  - Decision: n/a · Status: active

<!-- Future entries appended below after each measured iteration -->

---

## Rejected / Quarantine Log

Attempts that did not beat the best, or data that failed the gate. Kept for traceability; not in the active lineage.

| **Date** | **Item** | **Reason** | **Notes** |
|----------|----------|-----------|-----------|
| — | — | — | — |

---

## Versioning & Provenance

- **Brain version** increments on each **promoted** checkpoint: `vMAJOR.MINOR`.
- **Current brain version:** v1.0 (initialized)
- **Current best checkpoint:** base baseline (pending first FT)
- **Dataset version in use:** none yet
- **Iterations:** 0 promoted · 0 rejected

Every served model records the **brain version** it corresponds to — so any answer is traceable to an exact dataset + checkpoint + eval report.

---

## How This Makes LeanCoder Stronger Over Time

- **Failure-driven:** real wrong answers become future training examples (after verification) — the model learns from its own gaps.
- **Monotonic:** the promotion gate guarantees the "best" model only ever gets better on the eval suite.
- **Auditable:** every improvement is a versioned, benchmarked, reproducible step — not a black box.

> **Guardrail:** This brain governs *data + checkpoint lineage only*. It never bypasses the eval gate, never trains on held-out data, and never promotes on subjective judgment.
