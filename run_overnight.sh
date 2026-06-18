#!/usr/bin/env bash
# run_overnight.sh — psi-stories sub-1M capability-per-bit campaign.
#
# Trains the flagship sub-1M model + a size sweep + a multi-head ablation on TinyStories, evaluates
# each against the bar (generates completions for later grading), and logs everything to results/.
# Built to run UNATTENDED overnight. Launch when the Mac is free:   bash run_overnight.sh
#
# Priority order: the most important runs are first, so a partial night still yields the key results.

cd "$(dirname "$0")"
TS=$(date +%Y%m%d_%H%M%S)
OUT="results/overnight_$TS"
mkdir -p "$OUT"
exec > >(tee "$OUT/campaign.log") 2>&1
echo "=== psi-stories sub-1M campaign  $TS ==="

# --- data: a ~100 MB slice of the full TinyStories train set (better than the 21MB valid slice) ---
DATA="data/tinystories-train-100mb.txt"
if [ ! -s "$DATA" ]; then
  echo "downloading ~100MB train slice ..."
  curl -sL -r 0-104857600 -o "$DATA" \
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt" \
    || cp data/tinystories-valid.txt "$DATA"
fi
echo "data: $DATA  ($(ls -lh "$DATA" | awk '{print $5}'))"

# --- build fresh ---
echo "building psi_stories ..."
clang++ -std=c++17 -O3 -march=native -ffast-math -DPSI_REAL=float \
  src/step2_psi_nano/stories.cpp src/step3_metal/metal_backend.mm \
  -framework Metal -framework Foundation -o psi_stories || { echo "BUILD FAILED"; exit 1; }

run() { # name steps vocab d layers block hidden heads
  local name=$1 steps=$2 vocab=$3 d=$4 layers=$5 block=$6 hidden=$7 heads=$8
  echo ""
  echo "===== $name : steps=$steps vocab=$vocab d=$d L=$layers ctx=$block hid=$hidden heads=$heads  ($(date +%H:%M:%S)) ====="
  ./psi_stories train "$DATA" "$steps" "$vocab" "$d" "$layers" "$block" "$hidden" "$heads" \
    > "$OUT/$name.train.log" 2>&1 || { echo "$name TRAIN FAILED"; return; }
  cp psi_stories.bin "$OUT/$name.bin"
  local p last
  p=$(grep -o 'params=[0-9]*' "$OUT/$name.train.log" | head -1)
  last=$(grep '^step' "$OUT/$name.train.log" | tail -1)
  echo "$name | $p | $last" | tee -a "$OUT/SUMMARY.txt"
  ./psi_stories eval "$OUT/$name.bin" eval/tinystories_prompts.txt 0.7 \
    > "$OUT/$name.eval.log" 2>&1 || echo "$name EVAL FAILED"
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
echo "=== CAMPAIGN COMPLETE  $(date +%Y-%m-%d_%H:%M:%S) ===" | tee "$OUT/STATUS"
echo "results in $OUT/  (per-run .train.log/.eval.log/.bin; SUMMARY.txt; campaign.log)"
