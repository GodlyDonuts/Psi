# Psi — Optimization Log

Measured speedups to the custom CPU stack (the Mac prototype). **Rule: every entry keeps all
grad-checks passing — correctness is never traded for speed.** When a change would break a grad-check,
it is reverted.

**Benchmark:** `matmul_bench` — a 256×256×256 matmul, forward + both backward passes, reported in
GFLOP/s. Hardware: Apple Silicon, 8 cores (4 performance). matmul dominates the whole stack's cost,
so this number is the scoreboard.

| # | Change | GFLOP/s | vs baseline | grad-checks |
|---|---|---:|---:|---|
| 0 | baseline — naive `i-j-l` matmul, `-O2` | 1.80 | 1.0× | ✅ PASS |
| 1 | `i-l-j` loop reorder (contiguous inner loop → cache-friendly + vectorizable) + `-O3 -march=native` | 5.40 | **3.0×** | ✅ PASS |
| 2 | multithread matmul over disjoint output rows (4 perf cores; large matmuls only) | 10.95 (256³) / 14.24 (512³) | **6.1× / 7.9×** | ✅ PASS |

**psi-nano (the real workload):** step time fell **~3×** across iters 1–3.
- Iters 1–2 (reorder + flags; its small matmuls stay serial): step-100 5.9s → 2.6s, loss
  bit-identical (2.1949) — determinism preserved.
- Iter 3 — **float32 training path** (`-DPSI_REAL=float`); the grad-check oracle stays `double`, so
  correctness is still gated tightly. A 200-step run drops 7.3s → 5.1s (**~1.4×**) at equivalent loss
  (~1.83). All grad-checks still PASS.

**Iter 4 — `k`-unroll-by-4 register blocking (matmul forward): REVERTED.** Same-session A/B: 256³
25.8 → 21.1 GFLOP/s, 512³ 22.6 → 18.0 — a ~15–20% **regression**. Grad-checks passed (correctness was
fine), but it's slower: the compiler already auto-vectorizes the simple AXPY well, and manual unrolling
added register/cache pressure. Reverted per the rule — *no measured win → revert*.

⚠️ **Benchmark variance:** absolute GFLOP/s drift ~2× with machine load (the iter-2 "10.95" was under
load; the quiet-machine baseline for the same code is ~22–26). Only **same-session A/B** numbers are
trustworthy; the relative speedups in the table were same-session and stand.

**Status:** the simple vectorized matmul is at the practical CPU double-precision roofline (~22–26
GFLOP/s on 4 cores) — further matmul micro-opt shows diminishing/negative returns. Remaining clean CPU
wins live in **per-op overhead** (arena/tape autograd for psi-nano's many small ops) and a **persistent
thread pool**; the real 10–100× is **Step 3 — Metal GPU kernels** (best done with the user awake).

**Iter 5 — `-ffast-math` on the training build (psi-nano only): ~2.0× win, KEPT.** Same-session A/B
(200 steps): 3.6s/3.5s → 1.8s/1.7s with **identical loss (1.8250)** — convergence unchanged, no NaN.
The grad-check oracle stays strict `double` (no fast-math), so the math remains proven; fast-math only
relaxes FP reassociation/contraction on the training path, which is safe here (causal mask is −1e9 not
−Inf, softmax is max-subtracted, CE is guarded).

**Iter 6 — batch the projection matmuls (pack the batch into one `[B·T, d]` stream + block-diagonal
causal mask): correct but NEUTRAL, REVERTED.** The batched forward is *bit-identical* to per-sequence
(equivalence test: `0.00e+00`), but same-session A/B showed no speedup (1.8s → 1.8s): packing into one
`[256,256]` attention computes ~8× the cross-sequence score pairs (then masks them), cancelling the
gain from parallelizing the larger projection matmuls. Reverted.

**Key finding:** psi-nano's bottleneck is **not** matmul size — neither bigger matmuls (iter 6) nor
matmul micro-opt (iter 4) move its step time, while `-ffast-math` (iter 5) did. So the remaining cost
is **per-op / scalar overhead** (node allocation, graph build, the many small elementwise loops), not
GEMM. Next: profile to confirm, then **arena/tape autograd** to cut per-op allocation. The big leap
remains **Step 3 — Metal GPU kernels** (with the user).

**Iter 7 — profiling (added `psi_profile`; no engine change).** Phase breakdown of a psi-nano step
(8.6 ms/step): **forward+loss 39%, backward 61%, optimizer ~0%.** Confirms the diagnostic — the
optimizer (pure array math) is free; the cost is evaluating **~2000 tiny ops/step** (forward
node-build + compute, backward closure dispatch + grad loops). The matmuls are too small to be
throughput-bound, which is why iters 4 & 6 didn't move the needle. **No single micro-opt fixes this**
— the structural win is fewer/fused ops or a **tape/arena autograd** (build the graph once, replay it
each step), a deliberate core-autograd refactor best done carefully (with the user). The clean,
safe CPU quick-wins are now **exhausted**; the order-of-magnitude leap from here is **Step 3 — Metal
GPU kernels**.
