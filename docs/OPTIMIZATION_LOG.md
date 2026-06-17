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

_Next candidates: register/cache blocking (4×4 micro-kernel); arena/tape autograd to cut per-op
allocation in psi-nano; persistent thread pool (parallelize medium matmuls without per-call spawn);
Apple Accelerate/Metal as a roofline reference; then Step 3 — real Metal GPU kernels._
