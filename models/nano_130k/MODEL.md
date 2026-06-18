# psi-stories / nano_130k

A sub-1M **capability-per-bit** variation — part of the "smallest model that still clears the
TinyStories bar" search (see [docs/EVAL.md](../../docs/EVAL.md)).

| field | value |
|---|---|
| parameters | **131392** |
| tokenizer | small BPE, vocab = 512 |
| architecture | d=64 · layers=4 (uniq=2, block-shared) · heads=4 / kv=2 (GQA) · ctx=64 · hidden=192 (SwiGLU) · RoPE · RMSNorm · tied emb |
| training | 4000 steps · AdamW (1e-3, wd 0.01) · **WSD** schedule · batch 8 |
| data | TinyStories V2-GPT4 (~100 MB train slice) — see [data/README.md](../../data/README.md) |
| code version | git `d55c29d` |
| final loss | step  4000   train 2.6954   val 2.7718   (226.7s) |

## Reproduce
```sh
# 0. fetch the data — see data/README.md
# 1. build
clang++ -std=c++17 -O3 -march=native -ffast-math -DPSI_REAL=float \
  src/step2_psi_nano/stories.cpp src/step3_metal/metal_backend.mm \
  -framework Metal -framework Foundation -o psi_stories
# 2. train this exact config, then eval against the bar
./psi_stories train data/tinystories-valid.txt 4000 512 64 4 64 192 4 2 2
./psi_stories eval psi_stories.bin eval/tinystories_prompts.txt 0.7
```

## Files
- `train.txt` — full training curve (train + held-out val loss)
- `eval.txt` — completions on the capability-bar prompts (`eval/tinystories_prompts.txt`)
- `model.bin` — trained checkpoint (not committed; regenerate via Reproduce)

## Capability-bar grade (docs/EVAL.md rubric, graded by Claude)
Grammar **5** · Coherence **3** · Consistency **2** · Plot **2** · **clears bar? ❌ No**

_Graded by Claude (Opus 4.8). Remarkably TinyStories-flavored for 131K params + only 4000 steps (~4M
tokens, final loss 2.70): grammatical English with dialogue, named characters, the right vocabulary and
register ("Once upon a time", happy/sad/friends, `<|endoftext|>`). But it drifts — switches characters
mid-story, ignores the prompt's setup, no coherent arc. **Undertrained**: TinyStories coherence needs
loss ~1.5–2.0; we stopped at 2.70 due to the 8GB memory ceiling. A full run would reveal its true floor._
