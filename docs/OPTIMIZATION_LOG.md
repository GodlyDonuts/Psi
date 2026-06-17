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
thread pool**; the real 10–100× is **Step 3 — Metal GPU kernels** (best done with the user awake). Next
iteration targets the arena/tape autograd.
