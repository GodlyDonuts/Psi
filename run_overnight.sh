#!/usr/bin/env bash
# run_overnight.sh — psi-stories sub-1M capability-per-bit campaign.
#
# Trains the flagship sub-1M model + a size sweep + a multi-head ablation on TinyStories, evaluates
# each against the bar, and writes a SELF-CONTAINED, REPRODUCIBLE folder per model under models/<name>/:
#   MODEL.md  (config + exact reproduce command + git commit + final loss + grade)
#   train.txt (full training curve)   eval.txt (completions on the bar prompts)   model.bin (checkpoint)
# Built to run UNATTENDED overnight. Launch when the Mac is free:   bash run_overnight.sh
#
# Priority order: most important runs first, so a partial night still yields the key results.

cd "$(dirname "$0")"
TS=$(date +%Y%m%d_%H%M%S)
mkdir -p results
exec > >(tee "results/overnight_$TS.log") 2>&1     # live campaign log (gitignored)
GITCOMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)
echo "=== psi-stories sub-1M campaign  $TS  (code git $GITCOMMIT) ==="

# --- data: a ~100 MB slice of the full TinyStories train set (better than the 21MB valid slice) ---
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

run() { # name steps vocab d layers block hidden heads
  local name=$1 steps=$2 vocab=$3 d=$4 layers=$5 block=$6 hidden=$7 heads=$8
  local dir="models/$name"
  mkdir -p "$dir"
  local cmd="./psi_stories train $DATA $steps $vocab $d $layers $block $hidden $heads"
  echo ""
  echo "===== $name : steps=$steps vocab=$vocab d=$d L=$layers ctx=$block hid=$hidden heads=$heads  ($(date +%H:%M:%S)) ====="
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
| architecture | d=$d · layers=$layers · heads=$heads · context=$block · hidden=$hidden · RMSNorm · tied embeddings |
| training | $steps steps · AdamW (lr 1e-3, wd 0.01) · batch 8 |
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
- \`eval.txt\` — the model's completions on the capability-bar prompts (\`eval/tinystories_prompts.txt\`)
- \`model.bin\` — trained checkpoint (not committed — large binary; regenerate via Reproduce above)

## Capability-bar grade (docs/EVAL.md rubric, graded by Claude)
_Filled in after the run:_  Grammar — · Coherence — · Consistency — · Plot — · **clears bar? —**
EOF
  echo "$name | params=$params | $last"
}

# 1) flagship sub-1M (~965K) — the headline result
run flagship_965k  15000  1024 128 5 128 384 4
# 2) multi-head ablation at the same size (heads=1) — measures what multi-head bought us
run ablate_1head   15000  1024 128 5 128 384 1
# 3) size sweep downward — find where coherence breaks (the capability-per-bit frontier)
run mid_600k       10000  1024 96  5 128 320 4
run small_450k     10000  1024 96  4 96  256 4
run tiny_250k      10000  512  64  4 96  192 4
run nano_150k      10000  512  64  3 96  128 4

echo ""
echo "=== CAMPAIGN COMPLETE  $(date +%Y-%m-%d_%H:%M:%S) ===" | tee "models/CAMPAIGN_DONE_$TS"
echo "per-model folders under models/  (MODEL.md + train.txt + eval.txt + model.bin)"
