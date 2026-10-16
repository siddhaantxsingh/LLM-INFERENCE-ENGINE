#!/usr/bin/env bash
# Benchmarks and checks on SmolLM2-135M-Instruct (GGUF), CPU only.
#   bench/run_gguf.sh OUTDIR   (needs tools/fetch_model.sh for both SmolLM2 files and a
#                               Python venv with numpy, gguf, tokenizers, jinja2, llama-cpp-python as $PY)
set -euo pipefail
out=${1:?outdir}; mkdir -p "$out"
PY=${PY:-python3}
# llama.cpp uses OpenMP; with busy-waiting threads it stalled for >15 min on the shared, oversubscribed VM
export OMP_WAIT_POLICY=PASSIVE
F16=/home/user/models/smollm2-135m-instruct-f16.gguf
Q80=/home/user/models/smollm2-135m-instruct-q8_0.gguf
TXT=bench/data/alice_ch1-3.txt
L=target/release/llm
cargo build --release -q
load() { echo "# $(date -u +%FT%TZ) load: $(cut -d' ' -f1-3 /proc/loadavg)"; }
{
  for run in 1 2 3; do
    load
    $L bench --model $F16 --text $TXT --requests 1 --prompt-tokens 64 --gen-tokens 64 --max-seqs-list 1 2>/dev/null | tr -d '\n '; echo
    $L bench --model $F16 --q8 --text $TXT --requests 1 --prompt-tokens 64 --gen-tokens 64 --max-seqs-list 1 2>/dev/null | tr -d '\n '; echo
    $PY -I validate/bench_llamacpp.py $F16 $TXT 64 64 4 1 2>/dev/null
    $PY -I validate/bench_llamacpp.py $Q80 $TXT 64 64 4 1 2>/dev/null
  done
} > "$out/single_stream.jsonl"
for w in f32 q8; do
  flag=""; [ $w = q8 ] && flag=--q8
  for run in 1 2 3; do
    load
    $L bench --model $F16 $flag --text $TXT --requests 8 --prompt-tokens 64 --gen-tokens 32 --max-seqs-list 1,8 2>/dev/null
  done > "$out/throughput_$w.txt"
done
{ load; $L prefix-bench --model $F16 --text $TXT --requests 6 --prefix-tokens 384 --suffix-tokens 16 --gen-tokens 8 2>/dev/null; } > "$out/prefix_cache.txt"
{ load; $L ppl --model $F16 --text $TXT --window 512; $L ppl --model $F16 --q8 --text $TXT --window 512; $L ppl --model $Q80 --text $TXT --window 512; } > "$out/ppl.jsonl" 2>/dev/null
{ load; $PY -I validate/compare.py --model $Q80 --llm $L --out "$out/validate_smollm2_q8_0.json" --ppl $TXT; } > "$out/validate_smollm2_q8_0.log" 2>&1
load > "$out/done.txt"
