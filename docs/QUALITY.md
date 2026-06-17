# Psi — Quality Bar (Definition of Done)

_What "good" means for every component, and how we verify it. Companion to
[RADICAL.md](RADICAL.md) (which is about being original) — this one is about not shipping
anything broken or lazy on the way there._

## The core distinction

**"Good" and "SoTA" are different things, enforced at different times.**

- **"Good"** (correct, stable, principled, legible) is enforced on **every component, now**.
  It is non-negotiable and verifiable on a laptop.
- **"SoTA"** (throughput, capability-per-bit) is a **measured, comparative** outcome earned
  at the kernel and model layers — you benchmark your way there, you don't sprinkle it on
  each file. Trying to make a CPU reference "SoTA-fast" is premature optimization
  (see RADICAL.md) and we don't do it.

## The bar

| Dimension | What "good" means | How we verify | When |
|---|---|---|---|
| **Correctness** | every op's gradient is exact | finite-difference grad-check vs the oracle | every op — **now** |
| **Numerical stability** | no avoidable overflow/NaN | log-sum-exp softmax, stable fused cross-entropy | every op — **now** |
| **Right algorithm** | the principled choice, not the lazy one | cross-entropy (not MSE) for classification; fan-in-scaled init; AdamW | every component — **now** |
| **Legibility** | reads like a reference; the math is visible | review; comments state each backward's derivative | every file — **now** |
| **Throughput (MFU)** | near the hardware roofline | benchmark vs roofline | kernel layer (Step 3+) — *measured* |
| **Capability-per-bit** | on/above the frontier | eval vs SmolLM2/Qwen at matched size | model layer — *measured* |

The **top four are enforced continuously**. The **bottom two are the SoTA bets** and only
become real once there are real FLOPs and a real model to measure — not on XOR or a CPU
prototype.

## The grad-check discipline (the load-bearing rule)

**No op is used by a model until its analytic gradient matches a finite-difference
numerical gradient.** This is what lets us separate "is the math right?" from "is training
stable?" — so we never debug both at once. Concretely, the prototype already caught this in
action: psi-nano's ops grad-checked clean, training then NaN'd, and because the gradients
were *proven* correct we knew instantly it was a learning-rate issue, not an engine bug.

Tolerance: `max |analytic − numerical| < 1e-6` in double precision (we typically see ~1e-11).

## Definition of done, per component

- **An op** is done when: it has a backward, it grad-checks (< 1e-6), it's numerically
  stable, and its derivative is documented inline.
- **A model** is done when: it forwards/backwards without NaN, end-to-end grad-checks on a
  tiny instance, overfits a tiny input (memorization sanity), and trains with falling loss
  on a real corpus.
- **A kernel** (later) is done when: it bit-matches the CPU oracle within tolerance, *and*
  beats the baseline on its throughput metric — both, or it doesn't ship.

## Verified so far

- **Step 0** (scalar autograd): grad-check PASS (~1e-9); XOR solved.
- **Step 1** (tensor autograd): 6/6 op grad-checks PASS (~1e-11); XOR re-solved.
- **Step 2** (transformer ops): 8/8 grad-checks PASS (~1e-11); psi-nano initial loss = ln(V)
  (correct uniform prior); loss descends on the corpus.
