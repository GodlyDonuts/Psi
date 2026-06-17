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

**psi-nano (the real workload):** step-100 wall-time 5.9s → 2.6s after iters 1–2 (**2.3× faster**); its
matmuls are below the threading threshold so they stay serial (no regression), and the loss is
bit-identical (2.1949) — numerical determinism preserved.

_Next candidates: register/cache blocking (e.g. 4×4 micro-kernel); float32 training path (≈2×
bandwidth); arena/tape autograd to cut per-op allocation in psi-nano; persistent thread pool (avoid
per-call spawn)._
