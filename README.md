# Ψ (Psi)

A personal challenge to build a **state-of-the-art Small Language Model** — as small as
possible while as intelligent as possible — on a **fully custom training stack**: our own
autograd engine, runtime, and compute kernels. No PyTorch.

## Why this exists

The model is the byproduct, not the point. The real goals:

- **Master the full stack** — understand every layer from first principles: autograd, kernels,
  training dynamics, scaling.
- **Produce something novel** — a genuinely new kernel / architecture / training idea, not a
  reproduction.
- **The challenge & craft** — build it end-to-end, the hard way, on purpose.

Explicitly *not* a goal: a reusable, general-purpose framework. We optimize for understanding and
originality, and we stay free to specialize, hard-code, and throw things away.

Success is measured by **depth and originality**, not leaderboard optics. A useful secondary metric
is **quality-per-parameter / quality-per-training-FLOP** — the Pareto frontier for our compute
class — since a solo builder can't out-token the labs.

## Constraints

- **Fully custom stack** — own autograd → runtime → kernels. The deliberate uniqueness lever, and
  the main scope risk. Mitigated by an always-working incremental build.
- **Compute trajectory** — Apple Silicon (Metal / MLX) for the dev loop *now* → NVIDIA **GH200**
  (Grace Hopper) for scale → a future **AWS GPU cluster**. The design splits cleanly into a
  dev loop and a scale-up.

## Status

Research-first: landscape sweep → design doc → build.

- [x] Project charter & goals — see [docs/00-charter.md](docs/00-charter.md)
- [x] Design doc — see [docs/DESIGN.md](docs/DESIGN.md)
- [ ] Decide target capability — *recommended:* reasoning/math specialist, made tiny via
  distillation + a ternary (≈1.58-bit) track (see DESIGN §2). Your call.
- [ ] Custom stack: scalar autograd → tensor autograd → naive GPT loop → fused kernels → multi-device
- [ ] First trained model + eval against size-matched baselines
