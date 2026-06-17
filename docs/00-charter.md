# Psi — Project Charter

_Last updated: 2026-06-16_

## One line

Build a state-of-the-art Small Language Model — as small as possible while as intelligent as
possible — on a fully custom training stack, as a personal challenge to master the field and
produce something genuinely novel.

## The point (true-north)

What would make this a success **even if the final model were only mediocre**:

1. **Master the full stack** — first-principles understanding of autograd, kernels, training
   dynamics, and scaling.
2. **Produce something novel** — a new kernel / architecture / training idea worth sharing, not a
   reproduction.
3. **The challenge & craft** — building it end-to-end, the hard way, on purpose.

Deliberately **rejected** as a goal: building a lasting, reusable, general-purpose framework. We do
not pay the abstraction tax. We specialize, hard-code, and discard freely. The codebase is a sharp
instrument aimed at one model, not a mini-PyTorch.

## Honest framing of "SoTA"

A solo builder cannot out-token the frontier labs (leading sub-2B models see multiple trillions of
tokens). So "SoTA" here does **not** mean beating everyone on every benchmark. It means one of:

- **SoTA-for-our-compute-class** — push the Pareto frontier of quality-per-parameter and
  quality-per-training-FLOP. Measurable, defensible, and directly helped by our custom kernels.
- **Domain specialty** — genuinely beat much larger models on one vertical.

Primary success metric: **depth and originality**. Secondary: the Pareto numbers above.

## Constraints

- **Fully custom stack** — our own autograd engine, runtime, and compute kernels. No PyTorch / HF
  Trainer for the core path.
- **Compute trajectory**
  - *Now:* Apple Silicon (M-series) — Metal / MLX dev loop.
  - *Scale:* NVIDIA GH200 (Grace Hopper) — CUDA with TMA, `wgmma`, FP8, FlashAttention-3,
    NVLink-C2C coherent memory.
  - *Later:* AWS GPU cluster — multi-node.
  - The design must split cleanly into the Apple dev loop and the Hopper/cluster scale-up, with
    kernels written to port across.

## Working agreement

Mode: **I (the assistant) implement; you steer and learn.** With these rules, so it serves mastery
and craft rather than producing a black box:

- **Nothing is a black box.** Every nontrivial piece is a readable, first-principles reference with
  the reasoning and math exposed inline. If a library one-liner would hide something worth
  understanding, we write it out longhand.
- **You steer every real decision.** Forks are surfaced with a recommendation; you make the call.
- **Novelty is first-class.** We actively hunt for places to do something original (most likely in
  the kernels, possibly the training method) and flag "conventional way vs riskier original idea."
- **Standing offer:** any signature piece you'd rather hand-write yourself (autograd core, a key
  kernel), say so and the assistant scaffolds around it instead.

## Open decisions

- **Target capability** — what the model should be good at. Candidates:
  1. General-purpose, push the quality-per-param Pareto.
  2. Domain specialist (e.g. code, math, a structured task) — most realistic path to a true "SoTA"
     claim.
  3. Reasoning-focused tiny model (distillation / RL on traces).
  To be decided once the research-backed design doc lands.

## Build philosophy — always-working incremental path

Never a big-bang framework. Each step produces something that runs end-to-end:

1. **Scalar autograd** (micrograd-class) — reverse-mode autodiff from scratch.
2. **Tensor autograd** — n-dimensional arrays, broadcasting, the ops a transformer needs.
3. **Naive GPT training loop** — a small model trains and loss goes down, correctness first.
4. **Real fused kernels** — attention, GEMM, fused norm+residual, fused optimizer, fused
   cross-entropy. Apple/Metal first, ported to Hopper CUDA.
5. **Scale-up** — multi-device, larger token budgets, the over-trained small regime.

## Status / next

- Charter committed (this document).
- Research landscape sweep running → produces `docs/DESIGN.md` (architecture, data, training,
  compression, reasoning, custom stack, kernels, eval, phased roadmap, target numbers).
- After the design doc: decide target capability, then begin step 1 of the build.
