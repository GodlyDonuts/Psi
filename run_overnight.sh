#!/usr/bin/env bash
# run_overnight.sh — psi-stories sub-1M capability-per-bit campaign (modern architecture).
#
# Every model uses the full stack from docs/RESEARCH.md: small-BPE · multi-head + GQA · RoPE · SwiGLU ·
# block-wise weight-sharing (deep-and-thin) · tied embeddings · WSD LR schedule. Trains a frontier sweep
# on TinyStories, evals each against the bar, and writes a SELF-CONTAINED REPRODUCIBLE folder per model:
#   models/<name>/  =  MODEL.md (config + exact reproduce cmd + git commit + grade) · train.txt · eval.txt · model.bin
# Built to run UNATTENDED overnight. Launch when the Mac is free:   bash run_overnight.sh
# Priority order: flagship first (guaranteed headline), then shrink — a partial night still yields results.

cd "$(dirname "$0")"
TS=$(date +%Y%m%d_%H%M%S)
mkdir -p results
exec > >(tee "results/overnight_$TS.log") 2>&1
GITCOMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)
echo "=== psi-stories sub-1M campaign  $TS  (code git $GITCOMMIT) ==="

DATA="data/tinystories-train-100mb.txt"
if [ ! -s "$DATA" ]; then
  echo "downloading ~100MB train slice ..."
  curl -sL -r 0-104857600 -o "$DATA" \
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt" \
    || cp data/tinystories-valid.txt "$DATA"
fi
echo "data: $DATA  ($(ls -lh "$DATA" | awk '{print $5}'))"

echo "building psi_stories ..."
clang++ -std=c++17 -O3 -march=native -ffast-math -DPSI_REAL=float \
  src/step2_psi_nano/stories.cpp src/step3_metal/metal_backend.mm \
  -framework Metal -framework Foundation -o psi_stories || { echo "BUILD FAILED"; exit 1; }

run() { # name steps vocab d layers block hidden heads nkv nuniq
  local name=$1 steps=$2 vocab=$3 d=$4 layers=$5 block=$6 hidden=$7 heads=$8 nkv=$9 nuniq=${10}
  local dir="models/$name"
  mkdir -p "$dir"
  local cmd="./psi_stories train $DATA $steps $vocab $d $layers $block $hidden $heads $nkv $nuniq"
  echo ""
  echo "===== $name : $cmd   ($(date +%H:%M:%S)) ====="
  $cmd > "$dir/train.txt" 2>&1 || { echo "$name TRAIN FAILED (see $dir/train.txt)"; return; }
  cp psi_stories.bin "$dir/model.bin"
  ./psi_stories eval "$dir/model.bin" eval/tinystories_prompts.txt 0.7 > "$dir/eval.txt" 2>&1 || echo "$name EVAL FAILED"

  local params last
  params=$(grep -o 'params=[0-9]*' "$dir/train.txt" | head -1 | cut -d= -f2)
  last=$(grep '^step' "$dir/train.txt" | tail -1)
  cat > "$dir/MODEL.md" <<EOF
# psi-stories / $name

A sub-1M **capability-per-bit** variation — part of the "smallest model that still clears the
TinyStories bar" search (see [docs/EVAL.md](../../docs/EVAL.md)).

| field | value |
|---|---|
| parameters | **$params** |
| tokenizer | small BPE, vocab = $vocab |
| architecture | d=$d · layers=$layers (uniq=$nuniq, block-shared) · heads=$heads / kv=$nkv (GQA) · ctx=$block · hidden=$hidden (SwiGLU) · RoPE · RMSNorm · tied emb |
| training | $steps steps · AdamW (1e-3, wd 0.01) · **WSD** schedule · batch 8 |
| data | TinyStories V2-GPT4 (~100 MB train slice) — see [data/README.md](../../data/README.md) |
| code version | git \`$GITCOMMIT\` |
| final loss | $last |

## Reproduce
\`\`\`sh
# 0. fetch the data — see data/README.md
# 1. build
clang++ -std=c++17 -O3 -march=native -ffast-math -DPSI_REAL=float \\
  src/step2_psi_nano/stories.cpp src/step3_metal/metal_backend.mm \\
  -framework Metal -framework Foundation -o psi_stories
# 2. train this exact config, then eval against the bar
$cmd
./psi_stories eval psi_stories.bin eval/tinystories_prompts.txt 0.7
\`\`\`

## Files
- \`train.txt\` — full training curve (train + held-out val loss)
- \`eval.txt\` — completions on the capability-bar prompts (\`eval/tinystories_prompts.txt\`)
- \`model.bin\` — trained checkpoint (not committed; regenerate via Reproduce)

## Capability-bar grade (docs/EVAL.md rubric, graded by Claude)
_Filled in after the run:_  Grammar — · Coherence — · Consistency — · Plot — · **clears bar? —**
EOF
  echo "$name | params=$params | $last"
}

#    name           steps  vocab  d  layers ctx hidden heads nkv nuniq   (≈params)
run flagship_900k   15000  1024  128   8    128  384     4    2    4   #  ~918K — 8 deep / 4 shared
run mid_570k        12000  1024  128   6    128  256     4    2    3   #  ~574K
run small_350k      12000   512   96   6    128  256     4    2    3   #  ~354K
run tiny_215k       10000   512   96   4    128  192     4    2    2   #  ~215K
run nano_130k       10000   512   64   4    128  192     4    2    2   #  ~131K — how low can it still tell a story?

echo ""
echo "=== CAMPAIGN COMPLETE  $(date +%Y-%m-%d_%H:%M:%S) ===" | tee "models/CAMPAIGN_DONE_$TS"
echo "per-model folders under models/  (MODEL.md + train.txt + eval.txt + model.bin)"
