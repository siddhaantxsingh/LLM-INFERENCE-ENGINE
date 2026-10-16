# Benchmarks

## SmolLM2-135M-Instruct (GGUF, pretrained)

Produced by `bench/run_gguf.sh` and the commands in it, on the same 4-vCPU VM, 4 threads for both engines. **The machine was shared with other agents' builds and simulations**; the 1-minute load average is recorded before each run (in the result files). Two sets:

* `bench/results/2026-10-15/` (the full script): load average 12 to 25, i.e. 3 to 6 runnable processes per vCPU. Timings are heavily depressed and noisy.
* `bench/results/2026-10-15/quiet/` (single-stream and throughput only, rerun when the machine was quieter): load average 2.7 rising to 12 during the runs.

Treat all speeds as rough. Accuracy numbers do not depend on load.

### Accuracy: perplexity and int8

8,192 tokens of `bench/data/alice_ch1-3.txt`, 512-token windows (`ppl.jsonl`, `validate_smollm2_q8_0.json`):

| weights | engine perplexity | llama.cpp perplexity | weights in memory |
|---|---|---|---|
| F16 file → f32 | 18.8685 | 18.8695 | 651 MB |
| F16 file → engine int8 (`--q8`, per-row) | 18.9171 (+0.26 %) | — | 249 MB |
| Q8_0 file → f32 | 18.9151 (+0.25 %) | 18.9603 | 651 MB |

* On the Q8_0 file the engine still matches the float64 NumPy reference (max |logit diff| ≤ 3.9e-5, 32-token greedy equal on 3/3 prompts). llama.cpp differs more (max |diff| up to 2.9, top-1 agreement 87–100 %, greedy equal on 1/3 prompts) because it also quantises activations to 8 bits in Q8_0 matmuls. Its perplexity is 0.24 % higher than the engine's on the same file.
* The engine's int8 costs about the same perplexity as the Q8_0 file's 32-element-block quantisation.

### Speed, single stream (`quiet/single_stream.jsonl`, 3 runs)

64-token prompt, 64 greedy tokens, one request.

| implementation | decode tokens/s (3 runs) | TTFT for 64 tokens, ms (3 runs) |
|---|---|---|
| engine, f32 weights | 22.8 / 23.9 / 15.1 | 785 / 308 / 690 |
| engine, `--q8` int8 weights | 44.8 / 41.3 / 29.0 | 408 / 420 / 810 |
| llama.cpp, F16 file | 21.3 / 19.7 / 24.0 | 1138 / 240 / 328 |
| llama.cpp, Q8_0 file | 22.9 / 27.4 / 23.9 | 558 / 249 / 344 |

* Decode at batch 1 is limited by memory bandwidth: every token reads all weights (651 MB in f32, 249 MB in int8). int8 roughly doubles the engine's decode speed, unlike for the tiny trained model whose weights fit in cache.
* **The llama.cpp numbers are a lower bound, not a fair best case.** They come from llama-cpp-python with one Python `eval` call and a logits copy per token, `OMP_WAIT_POLICY=PASSIVE` (with the default busy-waiting OpenMP threads, a run stalled for over 15 minutes on the oversubscribed VM and was killed), and a shared machine. `llama-bench` was not built. Do not read this table as "the engine is as fast as llama.cpp".

### Throughput: continuous batching (`quiet/throughput_*.txt`, 1 run each)

8 requests, 64-token prompts, 32 greedy tokens each, all submitted at once.

| weights | max_seqs | output tokens/s | total tokens/s | TTFT p50 | wall |
|---|---|---|---|---|---|
| f32 | 1 | 11.0 | 33.1 | 13.2 s | 23.2 s |
| f32 | 8 | 34.3 | 102.9 | 4.7 s | 7.5 s |
| int8 | 1 | 29.4 | 88.2 | 4.7 s | 8.7 s |
| int8 | 8 | 54.7 | 164.2 | 2.3 s | 4.7 s |

Batching 8 sequences per step raised output throughput 3.1× with f32 weights and 1.9× with int8. The heavily loaded full-script runs (3 runs each, `throughput_*.txt`) show the same direction: f32 6.2–10.4 → 10.2–19.3 output tokens/s; int8 15.2–16.7 → 20.7–22.2.

### Prefix caching (`prefix_cache.txt`)

6 requests sent one after another, each a shared 384-token prefix plus a different 16-token suffix, 8 tokens generated (load average about 12):

| prefix cache | prompt tokens computed | TTFT, first request | TTFT, mean of the other 5 | wall |
|---|---|---|---|---|
| off | 2,400 | 9.9 s | 9.0 s | 59.2 s |
| on | 480 | 9.2 s | 0.36 s | 14.1 s |

With the cache, 1,920 of the 2,400 prompt tokens were served from cached blocks, and the time to first token after the first request fell about 25× (9.0 s → 0.36 s).

## The small model trained here

Produced by `bench/run.sh 2026-10-14` on a 4-vCPU cloud VM (Intel Xeon with AVX2/AVX-512; the engine uses its AVX2+FMA path), single process, 4 rayon threads. The model is the 3.0 M-parameter, 4-layer model trained in `train/` (see [design.md](design.md)); everything here is CPU.

## Accuracy

Held-out set: 59,136 tokens of Python standard-library files the model never saw, 256-token windows.

| evaluation | loss (nats/token) | perplexity | weight bytes |
|---|---|---|---|
| trainer (JAX, f32) | 1.2051337 | 3.3372 | — |
| engine, f32 | 1.2051338 | 3.3372 | 13.17 MB |
| engine, int8 weights | 1.2052032 | 3.3374 | 4.12 MB |

* The f32 engine and the JAX trainer differ by 1.8e-08 nats/token, which is float rounding. That is an end-to-end check of tokenizer, weights and forward pass at scale.
* int8 weights (per-row symmetric) cost 0.01 % perplexity. The int8 figure counts every weight matrix; the f32 embedding table used for input lookup stays alongside the quantised output projection.

## Throughput: continuous batching

64 requests, each with a 96-token prompt cut from held-out code, 128 greedy tokens each, all submitted at once. `max_seqs` caps how many sequences share a step.

| weights | max_seqs | output tokens/s | total tokens/s | wall time | mean decode batch | TTFT p50 / max |
|---|---|---|---|---|---|---|
| f32 | 1 | 610 | 1,068 | 13.4 s | 1.0 | 6.8 / 13.2 s |
| f32 | 4 | 1,169 | 2,046 | 7.0 s | 4.0 | 3.4 / 6.7 s |
| f32 | 16 | 1,744 | 3,052 | 4.7 s | 15.7 | 2.4 / 3.9 s |
| f32 | 64 | 1,883 | 3,296 | 4.3 s | 52.4 | 0.7 / 1.4 s |
| int8 | 1 | 616 | 1,078 | 13.3 s | 1.0 | 6.9 / 13.1 s |
| int8 | 16 | 1,586 | 2,775 | 5.2 s | 15.7 | 2.4 / 4.1 s |
| int8 | 64 | 1,816 | 3,179 | 4.5 s | 52.4 | 0.8 / 1.6 s |

## Speculative decoding

16 prompts (64 tokens), 128 greedy tokens each, one sequence at a time. The draft is the 1-layer, d = 128 model (held-out loss 1.79 vs 1.18). **Every speculative output was checked to equal plain greedy decoding** (the benchmark asserts it).

| mode | tokens/s | speedup | draft acceptance | target passes per token |
|---|---|---|---|---|
| plain greedy | 678 | 1.00× | — | 1.00 |
| speculative k=2 | 784 | 1.16× | 63 % | 0.45 |
| speculative k=4 | 714 | 1.05× | 52 % | 0.34 |
| speculative k=6 | 639 | 0.94× | 44 % | 0.29 |

## Reading the results

* **Batching is the main throughput lever.** Going from one sequence per step to 64 raises output throughput 3.1× (610 → 1,883 tokens/s). Each step reads every weight row once for all sequences, and per-step overheads (thread dispatch, small matmuls) are shared. The median time to first token falls with it, because queued requests start sooner.
* **int8 weights do not speed this model up.** All weights together are 13 MB and stay in this machine's cache, so the decode step is not limited by memory bandwidth, and converting int8 back to float costs extra instructions. Weight-only quantisation pays off when weights are much larger than the cache. Here it buys a 3.2× smaller footprint at no measurable loss in accuracy.
* **Speculative decoding gives a modest gain at small k.** The target needs fewer than half as many passes per token. But at batch 1 this model's step cost is dominated by fixed per-pass overhead, not by FLOPs, so the draft (about 10× fewer FLOPs) is not proportionally cheaper. Larger k loses because acceptance falls.

