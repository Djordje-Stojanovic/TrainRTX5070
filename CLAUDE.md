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

**Description column MUST be diagnostic.** Include: (1) what changed with values, (2) the hypothesis and evidence that motivated it, (3) what happened, (4) the conclusion — WHY it worked/failed and what this rules out. This is the AI's long-term memory. Future sessions read ONLY results.tsv, not git log.
- Bad: "WARMDOWN_RATIO 0.4→0.3"
- Bad: "reduce batch size for more steps"
- Good: "TOTAL_BATCH_SIZE 2^18→2^16 (hypothesis: 32 accum steps bottleneck MFU at 24%, evidence: web search MuonClip): val_bpb 2.14→2.05, MFU 24→31%, confirms accum was the bottleneck"
- Good: "WARMDOWN_RATIO 0.4→0.3 + constant WD (hypothesis: WD decay causes instability, evidence: exp#1 loss spike): loss still explodes 1.77→3.81, proves instability is NOT warmdown-related"
- Good: "DEPTH 12→14 (n_embd=896, ~220M) (hypothesis: 3.5GB VRAM headroom wasted, evidence: bottleneck diagnosis): val_bpb 1.15→1.10, more params utilizes VRAM headroom"

**Before proposing a new experiment, read results.tsv** to see what was already tried. Do not repeat a failed direction.

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

## Hardware Constraints

- **GPU:** RTX 5070, 12GB VRAM, Blackwell CC 12.0
- **Peak VRAM target:** <11.5 GB (96% of 12GB)
- **Autotune:** Automatically finds best device_batch_size + checkpointing combo, then **caches the result**. The cache key is GPU+PyTorch+seq_len — it does NOT include model size or TOTAL_BATCH_SIZE. So if you change model architecture/depth/width, the cached batch_size may be wrong.
- **After model size changes:** refresh autotune with `AUTORESEARCH_AUTOTUNE_REFRESH=1`
- **Autotune cache location:** `~\AppData\Local\autoresearch\gpu-profile-v3.json` — you can read this to see current training VRAM and tok/sec
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

## Bottleneck Diagnosis (check BEFORE planning next experiment)

After every experiment, check these metrics in order:

1. **VRAM utilization**: `peak_vram_mb` in the training output is EVAL vram, NOT training VRAM. Training VRAM is significantly higher. To check actual training VRAM, read the autotune cache: `cat ~/AppData/Local/autoresearch/gpu-profile-v3.json`. If training VRAM is well below 11.5GB, you have room for a bigger model.
2. **MFU**: Target 30%+ on this RTX 5070. Currently ~24%. Below 25% means throughput problem — too much gradient accumulation, bad batch size, or inefficient ops.
3. **Training stability**: If loss spikes or explodes, fix that before anything else. Search the web for current best practices with your optimizer.
4. **Loss curve shape**: If loss plateaus early, model likely needs more capacity or different architecture, not hyperparameter tuning.

**Fix bottlenecks in order. Never tune hyperparameters if training is unstable.**

**Hypothesis protocol (mandatory):** Every commit message must include: `Bottleneck: [X]. Hypothesis: [Y] because [Z]. Evidence: [prior experiment / web search / metric].` If you have no evidence, search the web first. After reverting a failed experiment, re-read train.py to confirm what state you're in — reverts undo ALL changes from that experiment, not just the one you're thinking about.

## Web Search — Your Most Powerful Tool

Don't just search when stuck. **Search proactively.** The field moves monthly. Today's date is March 2026.

**When to search:**
- Before your first experiment each session
- Every 5th experiment: "landscape scan" — pick 2-3 areas you haven't searched recently
- After 2-3 failures in the same direction
- Before any major change to a component you haven't researched yet
- Whenever a metric (MFU, VRAM, val_bpb) plateaus

**Search areas** (rotate through ALL of these over time — don't fixate on one):
1. **Architecture** — what's the current best architecture at this scale?
2. **Optimizer** — what's the current best optimizer and schedule?
3. **Memory efficiency** — how do SOTA models minimize VRAM per parameter?
4. **Throughput / MFU** — how do SOTA models maximize useful compute?
5. **Hardware-specific** — what does our GPU (Blackwell/CC 12.0) support that we're not using?
6. **Training techniques** — what training tricks are current SOTA models using?
7. **Leaderboard** — what are nanochat/modded-nanogpt leaders doing differently?
8. **Frontier** — what new techniques appeared in the last 3 months?

**How to search** (templates, not fixed strings):
- `"best [area] for small LLM training 2026"` — discover what's current
- `"[our current choice] vs alternatives 2026"` — question every assumption
- `"state of the art [area] March 2026"` — find what just changed
- `"nanochat autoresearch improvements"` — see what other agents found

**Rules:**
- Always add the current year. A 2023 technique may be obsolete.
- Never repeat the same search string twice across experiments — vary your queries.
- Read actual papers and code, not blog summaries.
- Question everything in train.py: is each component still SOTA, or was it SOTA in 2025?

## Context Window Management

Your context is finite. To maximize experiments per session:
- **Never read whole files** — use `grep -n "PATTERN" train.py` instead of reading the whole thing
- **Never cat run.log** — always use `grep` for specific metrics
- After 8+ experiments, re-read `results.tsv` to refresh what was tried
- Keep commit messages informative — they're your notes for after context truncation

## Tips for Good Experiments

- Make one change at a time when possible — easier to attribute improvements
- If val_bpb doesn't improve, revert (don't accumulate neutral changes)
- MFU matters: more compute per second = more learning per experiment
- Check `peak_vram_mb` — leaving VRAM headroom means you could be using a bigger model
- With 20-min budget, you get ~200-400 optimizer steps — enough for real learning dynamics
- Simpler is better at equal performance (see program.md simplicity criterion)
