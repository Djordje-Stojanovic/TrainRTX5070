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

## Continuing a Run

If you are dropped into this repo on an `autoresearch/*` branch with results already in `results.tsv`, **you are resuming an existing experiment loop.** Do NOT re-run setup. Just:

1. Read `CLAUDE.md` and `program.md` for context.
2. Read `results.tsv` to see what's been tried and the current best val_bpb.
3. Read `train.py` and `prepare.py` for the current code state.
4. Continue the experiment loop from where it left off.

## Logging Protocol

After EVERY experiment (keep, discard, or crash):

1. **Append the result to `results.tsv`** (never skip this, even for crashes).
2. **Commit `results.tsv`** after every experiment — if the session dies, results are preserved.
3. **Update `progress.png`** by running: `uv run plot_results.py --save` (silent, no GUI).

**Description column MUST be specific, quantitative, and diagnostic.** The description is how future sessions know what was already tried AND why it failed or succeeded. Include three things: (1) what changed (before→after values), (2) what happened (key metrics), (3) why (the diagnostic insight).

- Bad: "tweak hyperparams"
- Bad: "WARMDOWN_RATIO 0.4→0.3" (missing what happened and why)
- Good: "WARMDOWN_RATIO 0.4→0.3 + constant WD: loss still explodes 1.77→3.81 during cooldown, Muon momentum incompatible with LR decay"
- Good: "DEPTH 12→14 (n_embd=896, ~220M): val_bpb 1.15→1.10, more params utilizes VRAM headroom"

This prevents future sessions from repeating a failed DIRECTION (not just a failed value). If warmdown is fundamentally broken with Muon, say so — don't just say "0.3 didn't work" because the next AI will try 0.2.

**Before proposing a new experiment, read results.tsv** to see what was already tried. Do not repeat a failed experiment. If a direction was tried and failed, try a different direction.

**Push to GitHub after every experiment** so progress is visible remotely: `git push origin autoresearch/mar10`.

This ensures the human can always see what happened, even if the AI session crashes mid-loop.

## Waiting for Training Runs (IMPORTANT — save context tokens)

Training takes ~25 min total (20 min training + ~5 min startup/compile/eval). **You MUST NOT poll progress repeatedly.** Every tool call wastes context tokens and shortens your session. The goal is to maximize experiments per session (target: 10+ hours overnight).

**Protocol:**
1. Run training in background: `uv run train.py > run.log 2>&1` (use `run_in_background`)
2. **`sleep 300`** (5 min) — use bash `sleep`, with `timeout: 310000`
3. Check: `grep "^val_bpb:" run.log 2>/dev/null || echo "NOT DONE"`
4. If not done, **`sleep 300`** again. Repeat until done.
5. When done, extract all metrics with one grep.

This means ~5 checks per run, not 30+. Over a 10-hour session that's ~24 experiments × 5 checks = 120 tool calls for waiting, instead of 24 × 30 = 720.

**Calibrate sleep to run duration:** If you notice runs finish in ~22 min, sleep for the first 15 min (`sleep 900` with `timeout: 910000`), then check every 5 min. Always round down — better to sleep a bit less than miss completion by 1 second.

## Discarding Failed Experiments (CRITICAL)

**NEVER use `git reset --hard` to discard an experiment.** It destroys results.tsv entries, README changes, and other non-experiment files.

**Safe discard procedure:**
1. Append result to `results.tsv` and commit it.
2. Revert **only `train.py`** (and any other files you changed for the experiment): `git checkout <pre-experiment-commit> -- train.py`
3. Commit the revert.

This preserves all docs, results history, and other work. The failed experiment stays visible in results.tsv and the progress chart.

## When Stuck — Use Web Search

If you've tried 2-3 experiments in the same direction and nothing works, **search the web.** You have tools for this. Use them.

- Search for papers, blog posts, GitHub repos related to your problem
- Examples: "Muon optimizer learning rate schedule 2025", "SwiGLU warmdown instability", "efficient transformer training 12GB GPU 2026"
- Look at what nanochat leaderboard winners did: search "nanochat autoresearch improvements"
- Check recent ML papers on arXiv for architecture or optimizer innovations
- Read PyTorch docs for performance tips (fused kernels, memory-efficient ops)

**Always search for current SOTA (2025-2026).** Add the year to your searches. ML moves fast — a technique from 2023 may already be obsolete. Verify that any approach you find is still considered best practice as of 2026. Don't use deprecated APIs, abandoned libraries, or outdated architectures.

Don't waste 5 experiments guessing when a 10-second search could tell you the answer. Research first, then experiment.

## Discarding Experiments — NEVER use git reset --hard

When an experiment fails (val_bpb didn't improve), discard by reverting **only train.py**:

```bash
# CORRECT: revert only train.py to the pre-experiment state
git checkout <pre-experiment-commit> -- train.py
git commit -m "revert: undo <description>"

# WRONG: DO NOT DO THIS — it destroys results.tsv, README, and other files
git reset --hard <commit>
```

Always keep results.tsv, CLAUDE.md, README.md, and progress.png intact across discards.

## Tips for Good Experiments

- Always run the baseline first before changing anything
- Make one change at a time when possible — easier to attribute improvements
- If val_bpb doesn't improve, revert (don't accumulate neutral changes)
- MFU matters: more compute per second = more learning per experiment
- Check `peak_vram_mb` — leaving VRAM headroom means you could be using a bigger model
- With 20-min budget, you get ~200-400 optimizer steps — enough for real learning dynamics
- Simpler is better at equal performance (see program.md simplicity criterion)
