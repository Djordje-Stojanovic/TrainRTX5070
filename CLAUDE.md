# CLAUDE.md — Project Context for AI Agents

## MANDATORY RULES (read these FIRST, violating ANY is a critical bug)

1. **NEVER `git reset --hard`** — to discard, revert only train.py: `git checkout <commit> -- train.py`
2. **NEVER poll training** — use `sleep 300` (timeout: 310000) between checks, max 5 checks per run
3. **ALWAYS follow the post-experiment checklist** (below) — no exceptions, no skipping steps
4. **ALWAYS push after every experiment** — `git push origin autoresearch/mar10`
5. **NEVER stop the loop** — run experiments forever until manually interrupted
6. **NEVER change fairness invariants** — TIME_BUDGET, MAX_SEQ_LEN, evaluate_bpb(), dataset/tokenizer

## Post-Experiment Checklist (execute EVERY time, in order)

```
# 1. Log result to results.tsv (even for crashes)
echo -e "<commit>\t<val_bpb>\t<mem_gb>\t<mfu>\t<tok_sec>\t<steps>\t<params_M>\t<batch>\t<final_loss>\t<status>\t<description>" >> results.tsv

# 2. Commit results
git add results.tsv && git commit -m "results: <status> <short description>"

# 3. Update chart
uv run plot_results.py --save
git add progress.png && git commit -m "chart: update progress.png"

# 4. If DISCARD: revert only train.py to pre-experiment state
git checkout <pre-experiment-commit> -- train.py
git commit -m "revert: undo <description>"

# 5. Push everything
git push origin autoresearch/mar10
```

**Description column MUST be diagnostic.** Include: (1) what changed with values, (2) what happened, (3) WHY it failed/succeeded.
- Bad: "WARMDOWN_RATIO 0.4→0.3"
- Good: "WARMDOWN_RATIO 0.4→0.3 + constant WD: loss still explodes 1.77→3.81, proves instability is NOT warmdown-related"

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

## Continuing a Run

If you are dropped into this repo on an `autoresearch/*` branch with results already in `results.tsv`, **you are resuming an existing experiment loop.** Do NOT re-run setup. Just:

1. Read `CLAUDE.md` and `program.md` for context.
2. Read `results.tsv` to see what's been tried and the current best val_bpb.
3. Read `train.py` and `prepare.py` for the current code state.
4. Continue the experiment loop from where it left off.

## Waiting for Training Runs (save context tokens)

Training takes ~25 min total (20 min training + ~5 min startup/compile/eval).

**Protocol:**
1. Run training in background: `uv run train.py > run.log 2>&1` (use `run_in_background`)
2. **`sleep 300`** (5 min) — use bash `sleep`, with `timeout: 310000`
3. Check: `grep "^val_bpb:" run.log 2>/dev/null || echo "NOT DONE"`
4. If not done, **`sleep 300`** again. Repeat until done.
5. When done, extract all metrics with one grep.

Max ~5 checks per run. Calibrate: if runs take ~22 min, first sleep can be 10 min.

## When Stuck — Use Web Search

If you've tried 2-3 experiments in the same direction and nothing works, **search the web.**

- Examples: "Muon optimizer learning rate schedule 2025", "efficient transformer training 12GB GPU 2026"
- Look at nanochat leaderboard winners: search "nanochat autoresearch improvements"
- **Always search for current SOTA (2025-2026).** Add the year to searches.

Don't waste 5 experiments guessing when a 10-second search could tell you the answer.

## Tips for Good Experiments

- Always run the baseline first before changing anything
- Make one change at a time when possible — easier to attribute improvements
- If val_bpb doesn't improve, revert (don't accumulate neutral changes)
- MFU matters: more compute per second = more learning per experiment
- Check `peak_vram_mb` — leaving VRAM headroom means you could be using a bigger model
- With 20-min budget, you get ~200-400 optimizer steps — enough for real learning dynamics
- Simpler is better at equal performance (see program.md simplicity criterion)
