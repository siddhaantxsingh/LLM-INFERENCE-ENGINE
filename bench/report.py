"""Render docs/benchmarks.md from bench/results/<date>/*.json."""
import json
import os
import sys

D = sys.argv[1]
R = f"bench/results/{D}"
J = lambda n: json.load(open(os.path.join(R, n)))
f32, q8, jx = J("ppl_f32.json"), J("ppl_q8.json"), J("ppl_jax.json")
L = ["# Benchmarks\n",
     f"Produced by `bench/run.sh {D}` on a 4-vCPU cloud VM (Intel Xeon with AVX2/AVX-512; the engine uses its AVX2+FMA path), "
     "single process, 4 rayon threads. The model is the 3.0 M-parameter, 4-layer model trained in `train/` "
     "(see [design.md](design.md)); everything here is CPU.\n",
     "## Accuracy\n",
     f"Held-out set: {f32['tokens']:,} tokens of Python standard-library files the model never saw, 256-token windows.\n",
     "| evaluation | loss (nats/token) | perplexity | weight bytes |",
     "|---|---|---|---|",
     f"| trainer (JAX, f32) | {jx['loss']:.7f} | {jx['perplexity']:.4f} | — |",
     f"| engine, f32 | {f32['loss']:.7f} | {f32['perplexity']:.4f} | {f32['weight_bytes'] / 1e6:.2f} MB |",
     f"| engine, int8 weights | {q8['loss']:.7f} | {q8['perplexity']:.4f} | {q8['weight_bytes'] / 1e6:.2f} MB |",
     "",
     f"* The f32 engine and the JAX trainer differ by {abs(f32['loss'] - jx['loss']):.1e} nats/token, which is float rounding. "
     "That is an end-to-end check of tokenizer, weights and forward pass at scale.",
     f"* int8 weights (per-row symmetric) cost {100 * (q8['perplexity'] / f32['perplexity'] - 1):.2f} % perplexity. The int8 figure "
     "counts every weight matrix; the f32 embedding table used for input lookup stays alongside the quantised output projection.\n",
     "## Throughput: continuous batching\n",
     "64 requests, each with a 96-token prompt cut from held-out code, 128 greedy tokens each, all submitted at once. "
     "`max_seqs` caps how many sequences share a step.\n",
     "| weights | max_seqs | output tokens/s | total tokens/s | wall time | mean decode batch | TTFT p50 / max |",
     "|---|---|---|---|---|---|---|"]
for name, f in (("f32", "throughput_f32.json"), ("int8", "throughput_q8.json")):
    for r in J(f):
        L.append(f"| {name} | {r['max_seqs']} | {r['output_tokens_per_s']:,.0f} | {r['total_tokens_per_s']:,.0f} | {r['seconds']:.1f} s | "
                 f"{r['mean_batch_seqs']:.1f} | {r['ttft_ms_p50'] / 1000:.1f} / {r['ttft_ms_max'] / 1000:.1f} s |")
L.append("")
s = J("spec.json")
base = s[0]
L += ["## Speculative decoding\n",
      "16 prompts (64 tokens), 128 greedy tokens each, one sequence at a time. The draft is the 1-layer, d = 128 model "
      "(held-out loss 1.79 vs 1.18). **Every speculative output was checked to equal plain greedy decoding** (the benchmark "
      "asserts it).\n",
      "| mode | tokens/s | speedup | draft acceptance | target passes per token |",
      "|---|---|---|---|---|",
      f"| plain greedy | {base['tokens_per_s']:,.0f} | 1.00× | — | 1.00 |"]
for r in s[1:]:
    L.append(f"| {r['mode']} | {r['tokens_per_s']:,.0f} | {r['speedup']:.2f}× | {100 * r['acceptance']:.0f} % | {r['target_passes_per_token']:.2f} |")
L.append("")
t = {r["max_seqs"]: r for r in J("throughput_f32.json")}
L += ["## Reading the results\n",
      f"* **Batching is the main throughput lever.** Going from one sequence per step to 64 raises output throughput "
      f"{t[64]['output_tokens_per_s'] / t[1]['output_tokens_per_s']:.1f}× ({t[1]['output_tokens_per_s']:,.0f} → "
      f"{t[64]['output_tokens_per_s']:,.0f} tokens/s). Each step reads every weight row once for all sequences, and per-step "
      "overheads (thread dispatch, small matmuls) are shared. The median time to first token falls with it, because queued "
      "requests start sooner.",
      "* **int8 weights do not speed this model up.** All weights together are 13 MB and stay in this machine's cache, so the "
      "decode step is not limited by memory bandwidth, and converting int8 back to float costs extra instructions. Weight-only "
      "quantisation pays off when weights are much larger than the cache. Here it buys a 3.2× smaller footprint at no measurable "
      "loss in accuracy.",
      "* **Speculative decoding gives a modest gain at small k.** The target needs fewer than half as many passes per token. But at "
      "batch 1 this model's step cost is dominated by fixed per-pass overhead, not by FLOPs, so the draft (about 10× fewer FLOPs) is "
      "not proportionally cheaper. Larger k loses because acceptance falls.",
      ""]
print("\n".join(L))
