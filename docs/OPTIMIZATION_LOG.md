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

_Next candidates: multithread matmul over output rows (≈4 perf cores); register/cache blocking;
float32 path for the training loop; arena/tape autograd to cut allocation overhead._
