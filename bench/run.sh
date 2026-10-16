#!/bin/sh
# Reproduce docs/benchmarks.md: bench/run.sh [DATE]
set -eu
D=${1:-$(date +%Y-%m-%d)}
OUT=bench/results/$D
mkdir -p "$OUT"
cargo build --release
L=target/release/llm
python3 train/heldout.py "$OUT/heldout.txt" >/dev/null
# held-out loss: engine f32 and int8, and the trainer's own evaluation of the same windows
$L ppl --text "$OUT/heldout.txt" > "$OUT/ppl_f32.json"
$L ppl --text "$OUT/heldout.txt" --q8 > "$OUT/ppl_q8.json"
(cd train && python3 eval_ref.py artifacts/small.pkl "../$OUT/heldout.txt") > "$OUT/ppl_jax.json"
# offline throughput vs batch size
$L bench --text "$OUT/heldout.txt" --requests 64 --prompt-tokens 96 --gen-tokens 128 --max-seqs-list 1,4,16,64 > "$OUT/throughput_f32.json"
$L bench --text "$OUT/heldout.txt" --requests 64 --prompt-tokens 96 --gen-tokens 128 --max-seqs-list 1,16,64 --q8 > "$OUT/throughput_q8.json"
# speculative decoding with the 1-layer draft
$L spec --text "$OUT/heldout.txt" --requests 16 --prompt-tokens 64 --gen-tokens 128 --k-list 2,4,6 > "$OUT/spec.json"
rm "$OUT/heldout.txt"
python3 bench/report.py "$D" > docs/benchmarks.md
