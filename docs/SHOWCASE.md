# Psi — Showcase Strategy: a framework, not a model

_How Psi gets shown to the world. Resolves the long-open "what should it be good at?" decision._

## The reframe

The achievement is the **from-scratch training framework** — our own autograd and GPU kernels,
zero ML libraries — **not** any single tiny model. So the showcase is:

> **Psi — a from-scratch ML training framework (no PyTorch) + a zoo of tiny models you can try.**

A *family* of capable small models across different domains (`psi-stories` writes, `psi-chess` plays,
…) proves the framework is **general and real** — far stronger than one model, and honest to the
mission (the stack + the mastery were always the point — see [00-charter.md](00-charter.md),
[RADICAL.md](RADICAL.md)). It also turns "I trained a small model" (common) into "I built a framework
that trains capable models across domains" (rare, senior-signal).

## The bar: each model must not suck when a stranger tries it

A small (≤~30M, from-scratch) model has only enough capacity to be **fluent at one narrow thing.** So
"good" must mean *fluency / validity in a tight distribution* — never knowledge or reasoning. Pick
domains accordingly:

| Domain | Tiny model good at it? | Why |
|---|---|---|
| Coherent simple stories (TinyStories) | ✅ proven | narrow distribution; coherence is the bar and it's reachable at 5–15M |
| Game-move prediction (chess, Othello) | ✅ surprisingly | learns legal, club-level play; even tracks board state internally |
| Constrained short-form (names, words) | ✅ reliably | tiny models nail this; low ceiling but never embarrassing |
| Formal/structured (text→SQL, a DSL, regex) | ✅ verifiable | "good" = it parses/runs — hard to fake |
| Music (ABC / MIDI notation) | ✅ fun | plausible melodies you can play back |
| **Math / reasoning / Q&A / code / chat** | ❌ no | needs scale or a big-teacher distillation; a tiny from-scratch model will embarrass itself |

## The zoo — start narrow, then fan out

**Prove ONE flagship genuinely good first**, then add models (each is *cheap* once the framework +
kernels exist — just a new dataset + config; that's the payoff of building a framework).

- **`psi-stories`** — *flagship.* TinyStories; writes original, coherent short stories. Safest
  "won't suck," stays a true language model, and is literally the capability-per-bit benchmark.
- **`psi-chess`** — plays real chess, **legal-move-masked at inference** so it never blunders
  illegally; amateur-club strength. The interactive "wow" demo.
- **`psi-{names | sql | music}`** — a quick third for breadth.

**Anti-goal:** a pile of mediocre models. Two genuinely-good models beat six weak ones.

## Framework-ization — turning `psi-nano` (a script) into Psi (a framework)

The zoo forces the right engineering. Required:
- **Config** — size / layers / vocab / context as *data*, not hardcoded.
- **Tokenizer abstraction** — char-level (chess, names) + byte/BPE (stories).
- **Data pipeline** — read real datasets, train/val split, **held-out eval** (so we measure
  generalization, not memorization — psi-nano's current weakness).
- **Checkpoint save/load** — each trained model becomes a **shippable artifact** (no retrain-on-launch).
- **Clean `train` / `eval` / `generate` CLI.**

This generalization is itself showcase-worthy — it's the line between "a script" and "a framework."

## Plan

1. **Framework-ize `psi-nano`** — config, tokenizer abstraction, data pipeline, checkpoints.
   *(CPU-doable now; also fixes the hardcoded corpus + no-save-load limitations.)*
2. **GPU kernels** ✅ — Metal matmul tuned to **parity-class with MLX** (bit-exact; 99% of MLX on the
   key shape) **plus a novel ternary-weight GEMM** (16× smaller weights at full fp32-GEMM speed — the
   capability-per-bit flag, which MLX has no path for). Full journey + findings:
   [GPU_KERNELS.md](GPU_KERNELS.md). *Next: wire ternary into the autograd backend.*
3. **Flagship `psi-stories`** — prove it's genuinely good (held-out loss + novel samples).
4. **Fan out** — `psi-chess` + a quick third.
5. **Demo + writeup** — a web page where people pick a model and try it (full-stack skills tie the
   bow), plus the LinkedIn post.

## Honest constraints

- The **kernels gate** anything beyond toy scale.
- **Multi-week** effort; quality-per-model is non-negotiable.
- The **ternary track** ([RADICAL.md](RADICAL.md)) makes a zoo model "the smallest in MB that
  writes/plays X" — the capability-per-bit flag. The enabling kernel **already exists** (the ternary
  GEMM, [GPU_KERNELS.md](GPU_KERNELS.md)); a `psi-stories-ternary` is the natural headline demo.
