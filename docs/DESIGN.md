# Psi — Design Doc

_Drafted 2026-06-16 from first-principles knowledge of the 2024–2026 small-LM landscape.
Benchmark figures in §9 are from memory and **flagged for verification** before they're treated
as targets. This is a living document — we revise as we learn._

---

## 1. Executive summary & the honest framing of "SoTA"

**Goal restated:** as small as possible while as intelligent as possible, on a fully custom training
stack, with mastery + novelty + craft as the real prizes (not a reusable framework, not a résumé).

**The honest constraint.** A solo builder cannot win the absolute SLM crown by brute force. The
current leaders in the sub-2B class (SmolLM2-1.7B, Qwen2.5-1.5B, Llama-3.2-1B/3B) are trained on
**10–18 trillion tokens** with industrial data pipelines. You will not out-token them. So "SoTA"
here must mean one of two *defensible* things:

1. **SoTA-for-your-compute-class** — sit on or above the Pareto frontier of *quality per parameter*
   and *quality per training-FLOP*. Measurable, honest, and directly aided by your custom kernels.
2. **SoTA on a narrow axis** — be the smallest model (by **bits**, not just params) to clear a bar on
   a specific capability (e.g. grade-school + competition math). A claim you can actually own.

**The recommendation this doc builds toward:** a **small, reasoning-focused model specialized on
math (and optionally code), made tiny via distillation from a strong reasoning teacher, with a
ternary (≈1.58-bit) weight track as the novelty bet.** This is the tightest possible fit to all
three of your true-north goals *and* the custom-kernel ambition — explained in §2 and §6.

**Why this is a great project regardless of the final score:** the custom autograd + kernels force
first-principles mastery; ternary/low-bit training from scratch is genuinely under-explored
territory (novelty); and the whole thing is buildable end-to-end by one person on the
Apple → GH200 → AWS path (craft). The model is the byproduct; the understanding is the point.

---

## 2. Project framing & a recommendation on "what should it be good at?"

This was the deferred decision. Here are the three viable framings, then my pick.

| Framing | What you'd build | Can you credibly claim "SoTA"? | Fit to your goals |
|---|---|---|---|
| **A. General Pareto-pusher** | A broad ~0.3–1B base model, measured vs SmolLM2/Qwen at matched size | Only "good for compute class" — hard to stand out | Mastery: high · Novelty: low · easy to feel mediocre |
| **B. Domain specialist (non-reasoning)** | e.g. a tiny code-completion model | Yes, on the vertical | Mastery: high · Novelty: medium |
| **C. Reasoning specialist, made tiny** ⭐ | Small base → distilled reasoning (math/code) → ternary track | Yes — "smallest to hit X on MATH/GSM8K" | Mastery: high · Novelty: **high** · craft: high |

**My pick: C.** Rationale:

- **Reasoning distillation is the proven highest-ROI path to "intelligence" at tiny scale.** The
  existence proof is DeepSeek-R1-Distill-Qwen-1.5B — a 1.5B model that reaches competition-math
  numbers that looked impossible for the size, purely by SFT on reasoning traces from a big teacher.
  You don't need 11T tokens to be *smart at one thing*; you need good traces.
- **A domain focus makes "SoTA" attainable.** "Best general 0.5B" is a crowded, fuzzy claim. "Smallest
  model in bits to reach 80% on GSM8K" is sharp, novel, and yours.
- **"As small as possible" + "your own kernels" → ternary (BitNet-style 1.58-bit).** Ternary weights
  are the real frontier of *small*, they **must** be trained with custom kernels (you can't cleanly
  post-quantize into them), and a *ternary reasoning model* is largely unexplored. That is the
  novelty bet — and it's load-bearing on exactly the custom-kernel work you want to do anyway.

**Crisp success metric for C:** *capability-per-bit* — e.g. plot GSM8K/MATH accuracy against total
model size in **megabytes**, and be on the frontier. Secondary: quality-per-training-FLOP.

**Sequencing (respecting "always-working"):** prove the path in **bf16 first**, then chase ternary as
the differentiator. Don't combine two hard, novel things (custom stack + ternary) on the first run.

> **This is a recommendation, not a decision — your call.** If you'd rather keep it general (A) to
> maximize breadth of mastery, the stack, data, and kernel work below are ~90% identical; only the
> data mix and post-training change.

---

## 3. Recommended architecture

A decoder-only transformer with the modern small-model toolkit. Nothing exotic in the base — we save
the novelty budget for the kernels and the ternary track.

**Two sizes:**

- **`psi-nano` (~5–15M params)** — the *bring-up* model. Its only job is to overfit a tiny corpus and
  then train cleanly, so we can validate the custom autograd/kernels against a reference. Trains in
  minutes on an M-series Mac.
- **`psi-base` (~300–500M params)** — the real target. Big enough to be genuinely capable after
  reasoning distillation, small enough to train on a single GH200 in days. (Stretch to ~1B if compute
  is generous.)

**Architectural choices (and why):**

| Component | Choice | Why at small scale |
|---|---|---|
| Norm | **RMSNorm**, pre-norm | Cheaper than LayerNorm, standard, stable |
| Attention | **GQA** (e.g. 4–8 KV heads) + **QK-norm** | GQA cuts KV-cache; QK-norm stabilizes training (used in OLMo 2, Gemma 2) |
| Positional | **RoPE** (θ=10000, raise for long ctx) | De-facto standard; no learned pos params |
| MLP | **SwiGLU**, hidden ≈ (8/3)·d_model | Best quality/param for the FFN |
| Embeddings | **Tied input/output** | At 135M-scale, embeddings are ~20% of params — tying is free capability |
| Depth vs width | **Deep & thin** | MobileLLM's finding: at <1B, more layers beats more width at fixed params |
| Vocab | ~32k–49k BPE | Larger vocab = fewer tokens/text but more embedding params; moderate is right for tiny models |
| Activation/extras | optional ReLU² MLP, logit soft-cap | Cheap tricks from the nanoGPT speedrun worth ablating |

**Concrete `psi-base` starting point (to ablate, not gospel):** d_model 1024, 24–28 layers, 16 query
heads / 4 KV heads, SwiGLU hidden ~2730, vocab ~32k, ctx 2048 (extendable). That's ~0.35–0.45B.

**Novelty hooks to keep in view (don't do all):** DeepSeek-style **MLA** (low-rank KV compression),
**Muon**-friendly layer shapes, and the ternary-native design in §6.

**Tokenizer:** reuse an existing BPE (e.g. a tiktoken/HF tokenizer) to start — writing your own BPE
is a fun side-quest but not the point of the project. Revisit only if the domain (math) wants custom
tokens (digits, LaTeX).

---

## 4. Data strategy

Data quality dominates everything at this scale. The plan splits by phase.

**Pretraining (base capability):**
- Backbone web text: **FineWeb-Edu** (classifier-filtered educational CommonCrawl) and/or
  **DCLM-baseline** — both are strong, openly available, and already heavily filtered.
- Synthetic textbooks: **Cosmopedia v2** for the "Phi-style" clean explanatory data.
- Domain tilt (for framing C): heavy **math** (OpenWebMath, FineMath, InfiMM-WebMath) and **code**
  (Stack-Edu / The Stack v2) in the mix.
- Hygiene: MinHash dedup; decontaminate against eval sets (n-gram overlap) from day one.

**Token budget (be realistic):**
- `psi-nano`: ~0.1–1B tokens (bring-up only).
- `psi-base`: aim **100B–500B tokens**. That's far past Chinchilla-optimal (~20 tok/param) — which is
  correct, because small models are deliberately *over-trained* for deployment. It is also ~20–100×
  *less* than the SOTA leaders, which is why we lean on distillation and a narrow domain rather than
  raw breadth.

**Decay / "mid-training" phase:** with a WSD schedule (§5), the final LR-decay phase is where you
upweight your highest-quality data — instruction-like text, clean math, curated reasoning. This phase
gives an outsized quality jump; treat the data mix here as a tuned knob.

**Post-training data (the intelligence):** curated reasoning traces distilled from a strong teacher
(see §6). The s1/LIMO result — strong reasoning from ~1k *well-chosen* examples — means **trace
quality ≫ trace quantity**. Budget effort into curation, not volume.

---

## 5. Training recipe

**Regime:** over-trained / inference-optimal. Don't stop at compute-optimal; keep going while loss
improves, because inference cost is what we're optimizing for.

**Schedule:** **Warmup-Stable-Decay (WSD)**. Warmup (~1–2%), long stable phase at peak LR, then a
short sharp decay where you also swap in the high-quality data mix. WSD (vs cosine) lets you branch
continuation runs from the stable checkpoint and gives a clean place to do the data anneal.

**Optimizer:** start with **AdamW** (β=(0.9, 0.95), wd 0.1, grad-clip 1.0) for correctness. Then make
**Muon** a first-class experiment — it orthogonalizes momentum (Newton–Schulz) on 2D weight matrices
and posted real wall-clock wins on the nanoGPT speedruns; embeddings/norms stay on AdamW. Muon is both
a likely efficiency win *and* a novelty/kernel opportunity (the Newton–Schulz iteration is a fun
custom kernel). SOAP/Shampoo are second-order alternatives if you want to go deeper.

**Hyperparameter transfer:** adopt **µP (maximal-update parametrization)** so you can tune LR on a
tiny proxy and transfer to `psi-base` without re-sweeping. This is the single biggest cost-saver for a
solo builder — it turns "I can't afford to tune" into "I tuned on a model that trains in minutes."

**Precision:** **bf16** mixed-precision as the baseline everywhere. **FP8** is a later GH200-only
experiment (Hopper Tensor Cores support it) — powerful but finicky; not for bring-up.

**Stability kit:** z-loss on the softmax, QK-norm, careful scaled init, gradient clipping. Validate
every backward pass against finite-difference gradients on `psi-nano` before trusting the stack.

---

## 6. Making it smart: distillation, quantization, reasoning (priority order)

This is where a *small* model becomes *intelligent*. Ordered by ROI.

1. **Reasoning distillation (highest ROI).** SFT `psi-base` on chain-of-thought traces generated by a
   strong open reasoning teacher (e.g. a large DeepSeek-R1 / Qwen reasoning model) on math/code
   problems with verifiable answers. This is the lever that made R1-Distill-1.5B punch so far above
   its weight. Curate hard; quality ≫ quantity (s1/LIMO).
2. **Logit/hidden-state distillation during/after pretraining (optional, strong).** Llama-3.2-1B/3B
   and Gemma-2-2B were built with distillation from bigger siblings. If you can run a teacher's
   forward pass over your data, KL-matching its soft labels beats hard-label training.
3. **RL with verifiable rewards (GRPO), if compute allows.** After SFT, **GRPO** (DeepSeek's
   value-model-free PPO variant) on math with exact-match rewards can squeeze more reasoning out.
   Cheaper than PPO, but still the most compute-hungry step — treat as a stretch goal.
4. **Ternary / 1.58-bit weights — the novelty bet.** BitNet b1.58 showed you can train models with
   weights in {−1, 0, +1} *from scratch* and stay competitive, at a fraction of the bits. This is the
   "as small as possible" frontier and it **requires custom kernels** (ternary matmul / packed
   weights) — i.e. it's exactly the work you want to do. Plan: get bf16 `psi-base` working first, then
   train a ternary `psi-base-1.58` and compare on capability-per-bit. A *ternary reasoning model* is
   largely uncharted — that's the headline novelty.
5. **PTQ (AWQ/GPTQ to INT4)** as a pragmatic inference-time fallback for the bf16 model. Speculative
   decoding if you want faster inference (no capability change).

---

## 7. The custom training stack — always-working incremental build

The core of the project. Each step **runs end-to-end** before the next begins. Build vs reuse line:
**you hand-write the autograd, the model, the training loop, and the kernels** (the mastery + novelty
core); you may reuse a tokenizer and trivial file IO (not the point).

| Step | Deliverable | "Done" when… | Reference to study |
|---|---|---|---|
| **0. Scalar autograd** | micrograd-class reverse-mode autodiff + tiny MLP | it learns XOR; grads match finite-diff | Karpathy `micrograd` |
| **1. Tensor autograd** | ndarray + broadcasting + the ~20 ops a transformer needs (matmul, softmax, RMSNorm, SwiGLU, embedding gather, cross-entropy, RoPE) with backward | a `psi-nano` forward+backward matches a reference (e.g. MLX) within tolerance | tinygrad internals |
| **2. Naive training loop** | data loader, AdamW, checkpointing; train `psi-nano` | `psi-nano` overfits a tiny set, then trains on real data and val-loss drops | `nanoGPT` |
| **3. Real kernels** | fused Metal kernels replacing the slow ops (§8) | same loss curve, multiples faster, profiled | `llm.c`, FlashAttention |
| **4. Scale-up** | GH200 CUDA port, bigger data, µP transfer to `psi-base` | `psi-base` trains to target loss on GH200 | `llm.c` (CUDA), CUTLASS |
| **5. Novelty track** | Muon optimizer kernel; ternary matmul; `psi-base-1.58` | ternary model on the capability-per-bit frontier | BitNet, modded-nanoGPT |

**Autograd design:** a dynamic **tape** (record ops in forward, replay in reverse) over tensor ops is
the pragmatic core — simple, debuggable, and enough for a specialized model. Resist building a general
graph compiler; we explicitly rejected "reusable framework." Specialize and move.

**Reference implementations worth reading deeply:** `micrograd` (autodiff in ~150 lines), `nanoGPT`
(minimal real GPT), **`llm.c`** (GPT trained in raw C/CUDA — literally your end state, no framework),
`tinygrad` (lazy IR + codegen, if you ever want autogen kernels), and `modded-nanoGPT` (a treasure
chest of efficiency tricks: Muon, value embeddings, etc.).

---

## 8. Kernel roadmap: Apple Silicon (dev) → GH200 (scale) → AWS (multi-node)

**Which kernels matter** (by share of training time): **GEMM** (matmul — QKV/proj/MLP, dominates) →
**fused attention** (FlashAttention-style, never materialize the full score matrix) → **fused
cross-entropy** (fuse softmax+CE so you don't materialize the huge `[seq, vocab]` logits) → **fused
elementwise** (RMSNorm+residual, SwiGLU) → **fused optimizer step** (AdamW/Muon).

**Apple Silicon (now — dev & correctness):**
- Use **MLX** as the reference/baseline (unified-memory, Metal-backed, lazy) and write custom kernels
  in **Metal Shading Language**, callable via MLX's custom-kernel API or directly. MPS for optimized
  primitives where you just need a baseline.
- M-series is for *bring-up and correctness on small models*, not 100B-token runs — set expectations.
- Bonus: M-series **unified memory** is a conceptual rehearsal for GH200's coherent memory.

**NVIDIA GH200 / Hopper (scale):**
- GH200 = Grace ARM CPU + Hopper GPU over **NVLink-C2C** (~900 GB/s coherent) with large memory — big
  batches, cheap CPU offload, no PCIe wall. The GPU is Hopper, so: **Tensor Cores**, **`wgmma`**
  (async warpgroup MMA), **TMA** (async bulk copies), and **FP8**.
- Start GEMM on **cuBLAS/CUTLASS** (don't hand-write a competitive GEMM first); hand-write the
  **fused attention** (study **FlashAttention-3** — Hopper-specific, uses TMA+`wgmma`(+FP8) — and the
  tile-based **ThunderKittens** style for approachable Hopper kernels). Add FP8 last.
- **Portability rule:** keep the *math* identical between Metal and CUDA; only the backend swaps.
  Write each kernel with a CPU/MLX reference you can diff against on `psi-nano`.

**Triton** is the pragmatic middle layer for the Hopper side if hand-writing CUDA for everything is
too slow — but for *this* project (full custom, mastery), hand-written CUDA for the 2–3 hot kernels is
the higher-learning path; reuse libraries (cuBLAS/CUTLASS) only for the GEMM you're not trying to beat.

**AWS (later):** multi-node data/tensor parallelism. Out of scope until `psi-base` trains single-node.

---

## 9. Evaluation plan & target numbers

**Harness:** EleutherAI **lm-evaluation-harness** (and/or lighteval) for reproducibility. Pin prompt
format, few-shot count, and normalization (acc vs acc_norm) — these silently move numbers 5–10 points.

**Base-model suite:** MMLU (5-shot), ARC-easy/challenge, HellaSwag, WinoGrande, PIQA, OpenBookQA,
TriviaQA. **Reasoning suite (framing C):** GSM8K (8-shot CoT), MATH / MATH-500, and AIME-style for the
hard end; HumanEval/MBPP if code is in scope. IFEval for instruction-following after post-training.

**Pareto measurement (the real metric):** plot suite-average vs (a) **params** and (b) **training
FLOPs** (≈ 6·N·D) on log axes; overlay the public small models. For framing C, also plot
**capability vs model size in MB** (capability-per-bit) — the axis where ternary wins.

**Target numbers — ⚠️ FROM MEMORY, VERIFY BEFORE USING AS TARGETS.** Approximate base-model **MMLU
(5-shot)** and **GSM8K**, ordered by size, as rough goalposts:

| Model | Params | MMLU (approx) | GSM8K (approx) |
|---|---|---|---|
| random baseline | — | 25% | ~0% |
| SmolLM2-360M | 360M | ~35% | low |
| Qwen2.5-0.5B | 0.5B | ~47% | moderate |
| SmolLM2-1.7B | 1.7B | ~50% | ~30% |
| Gemma-2-2B | 2.6B | ~52% | moderate |
| Qwen2.5-1.5B | 1.5B | ~60% | strong (Qwen is math-heavy) |
| Llama-3.2-1B / 3B | 1B / 3B | ~37% / ~56% | low / moderate |
| **DeepSeek-R1-Distill-Qwen-1.5B** | 1.5B | — | **MATH-500 ~83%, AIME ~28%** |

The last row is the one that matters for framing C: a 1.5B *reasoning-distilled* model reaching
competition-math numbers is the existence proof that **a small Psi can be genuinely smart at math.**
Our `psi-base` target: land on or above the params↔capability frontier for math at ~0.3–0.5B, then
push the capability-per-bit frontier with the ternary track.

> Action item before we lock targets: re-pull these from each model's card / a current
> lm-eval-harness leaderboard. (This is exactly the fact-checking the research workflow was meant to
> do; we can run a hardened, scoped version of it later just for the numbers.)

---

## 10. Phased roadmap

| Phase | Goal | "Done" when… |
|---|---|---|
| **P0 — Scaffold** | Repo, charter, this doc, eval harness wired up | ✅ charter pushed; doc in repo |
| **P1 — Autograd** | Scalar → tensor reverse-mode autodiff (Step 0–1) | `psi-nano` fwd+bwd matches a reference within tol |
| **P2 — First training loop** | AdamW + WSD + data loader; train `psi-nano` (Step 2) | `psi-nano` overfits, then val-loss drops on real data |
| **P3 — Kernels (Apple)** | Fused attention, RMSNorm, cross-entropy, optimizer in Metal (Step 3) | same loss curve, profiled, multiples faster |
| **P4 — Scale (GH200)** | CUDA port; µP transfer; train **bf16 `psi-base`** on 100B+ tokens (Step 4) | `psi-base` hits base-model target numbers |
| **P5 — Intelligence** | Reasoning distillation SFT (+ optional GRPO) on math | `psi-base` lands on the math params↔capability frontier |
| **P6 — Novelty** | Muon kernel; **ternary `psi-base-1.58`**; capability-per-bit study (Step 5) | ternary model on/above the bits↔capability frontier |
| **P7 — Write it up** | Honest report: what worked, the kernels, the numbers | a post/paper others can learn from |

Apple Silicon carries P1–P3; GH200 carries P4–P6; AWS only if P6 wants multi-node scale.

---

## 11. Key open decisions & risks

**Open decisions (yours to make):**
1. **Target capability** — adopt framing C (reasoning/math + ternary), or A (general), or B (other
   domain)? Everything downstream (data mix, post-training) keys off this; the stack/kernels don't.
2. **`psi-base` size** — ~0.35B (safe on one GH200) vs ~1B (more capable, more compute).
3. **Tokenizer** — reuse existing vs custom (custom only earns its keep if math/LaTeX tokens help).
4. **How hard to commit to ternary** — headline novelty vs a stretch goal after bf16 works.

**Risks & mitigations:**
- **Scope blow-up of the custom stack** (the #1 risk). *Mitigation:* the always-working ladder in §7 —
  never more than one new layer un-validated; `psi-nano` + finite-diff as a constant correctness net.
- **Apple compute is too weak for real training.** *Accepted by design:* Apple = dev/correctness only;
  real runs are GH200/AWS. Don't try to train `psi-base` on a Mac.
- **Ternary + reasoning is two hard novelties at once.** *Mitigation:* bf16 first, ternary second.
- **Distillation depends on a good teacher + verifiable tasks.** *Mitigation:* pick math/code where
  answers are checkable; lean on s1/LIMO-style small, high-quality trace sets.
- **Benchmark numbers / "SoTA" overclaiming.** *Mitigation:* §9 numbers are flagged unverified; lock
  targets only after re-pulling from primary sources; claim "SoTA-for-class," not "SoTA," unless the
  narrow-axis claim is airtight.

---

### Immediate next step

If you adopt framing C (or even before deciding), **P1 is identical**: start the scalar→tensor
autograd. I can scaffold the repo layout and write Step 0 (scalar reverse-mode autodiff + a tiny MLP
that learns XOR, with a finite-difference gradient check) whenever you're ready.
