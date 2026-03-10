# CLAUDE.md — Project Context for AI Agents

## MANDATORY: Follow the Guides

**You MUST follow `program.md` exactly.** It defines the setup procedure, experiment loop, logging format, and operational rules. Do not skip steps, do not improvise the workflow. Read it first, follow it to the letter.

- **Before your first run:** Complete ALL setup steps in program.md (run tag, branch, read files, verify data, initialize results.tsv, confirm with user).
- **During experimentation:** Follow the experiment loop exactly (commit before run, log after run, keep/discard based on val_bpb, never stop).
- **Fairness invariants** below are non-negotiable.

## What is this?

Autonomous LLM pretraining research on a single RTX 5070 (12GB, Blackwell CC 12.0).
The AI agent runs experiments in a loop: modify code, train for 20 minutes, check if val_bpb improved, keep or discard, repeat.

Read **`program.md`** for the full experiment loop protocol, logging format, and operational rules.

## Architecture Overview

| Component | Current State |
|-----------|--------------|
| Model | SwiGLU MLP, RoPE, d12 (768 dim, 6 heads), ~162M params |
| Dataset | ClimbMix (nvidia/Nemotron-ClimbMix), pre-tokenized with GPT-2 |
| Tokenizer | GPT-2 (vocab=50257), EOT token as BOS |
| Optimizer | Muon (matrices) + AdamW (embeddings, scalars) |
| Compile | torch.compile via triton-windows |
| Attention | SDPA with is_causal=True (FlashAttention fast path) |
| Time budget | 20 minutes per experiment |
| Metric | val_bpb (bits per byte) — lower is better |

## File Map

```
CLAUDE.md       — YOU ARE HERE. Project context for AI agents.
program.md      — Experiment loop protocol. READ THIS FIRST.
train.py        — Model, optimizer, training loop. PRIMARY EDIT TARGET.
prepare.py      — Data pipeline, tokenizer, evaluation, constants.
pyproject.toml  — Dependencies.
results.tsv     — Experiment log (created during runs).
```

## What You Can Change

**`train.py` — primary edit target (anything goes):**
- Model architecture (layers, dimensions, attention, MLP, embeddings)
- Optimizer (learning rates, schedules, weight decay, momentum)
- Hyperparameters (batch size, depth, aspect ratio, warmup, cooldown)
- Training loop logic (gradient accumulation, loss scaling, etc.)

**`prepare.py` — allowed but with constraints:**
- You may modify the dataloader for efficiency (packing, prefetching, etc.)
- You may adjust `EVAL_TOKENS` within reason (must remain enough for reliable BPB)
- You **MUST NOT** change `evaluate_bpb()` — it is the ground truth metric
- You **MUST NOT** change `MAX_SEQ_LEN` — it anchors comparison fairness
- You **MUST NOT** change `TIME_BUDGET` — it anchors comparison fairness
- You **MUST NOT** change the tokenizer or vocab for ClimbMix (GPT-2, 50257)

**`pyproject.toml` — only if you genuinely need a new dep:**
- Adding a package is allowed if it enables a real optimization (e.g., a fused kernel)
- Do not add packages speculatively

## What You Must Not Change

These are the **fairness invariants** that make experiments comparable:

1. **`TIME_BUDGET = 1200`** (20 minutes) — the fixed training wall clock
2. **`MAX_SEQ_LEN = 2048`** — context length
3. **`evaluate_bpb()`** — the metric definition (nats per byte -> bits per byte)
4. **Dataset/tokenizer identity** — ClimbMix with GPT-2 tokenizer
5. **Evaluation data** — the val split must remain untouched

## Commands

```bash
# Prepare data (one-time)
uv run prepare.py --dataset climbmix

# Smoke test (~30s)
uv run train.py --smoke-test

# Full experiment (~20 min + startup/eval)
uv run train.py

# Run and capture output
uv run train.py > run.log 2>&1

# Check results
grep "^val_bpb:\|^peak_vram_mb:\|^mfu_percent:" run.log
```

## Hardware Constraints

- **GPU:** RTX 5070, 12GB VRAM, Blackwell CC 12.0
- **Peak VRAM target:** <11.5 GB (96% of 12GB)
- **Autotune:** Automatically finds best batch_size + checkpointing combo
- If OOM at all batch sizes, reduce model size or enable more aggressive checkpointing

## Tips for Good Experiments

- Always run the baseline first before changing anything
- Make one change at a time when possible — easier to attribute improvements
- If val_bpb doesn't improve, revert (don't accumulate neutral changes)
- MFU matters: more compute per second = more learning per experiment
- Check `peak_vram_mb` — leaving VRAM headroom means you could be using a bigger model
- With 20-min budget, you get ~200-400 optimizer steps — enough for real learning dynamics
- Simpler is better at equal performance (see program.md simplicity criterion)
