# Requirements matrix — Project 7 (LLM inference engine)

| Requirement | Status | Evidence |
|---|---|---|
| Correct transformer forward pass (RMSNorm, RoPE, GQA, SwiGLU) | VERIFIED | NumPy float64 reference; held-out loss = JAX to 1.8e-8 |
| Tokenizer | VERIFIED | byte-identical to the Python trainer's tokenizer on held-out text |
| Paged KV cache | VERIFIED | non-contiguous blocks in reference test; block size 4/16; NaN poisoning |
| Continuous batching, chunked prefill | VERIFIED | exact-equality tests; 3.1× throughput |
| Prefix caching | VERIFIED | reuse, position/prefix correctness, eviction tests |
| Preemption under KV pressure | VERIFIED | exact outputs with a 12-block pool |
| Sampling (greedy, temperature, top-k, top-p, seeds, stop strings) | VERIFIED | sampler distribution test; seed and stop tests |
| Weight quantisation | VERIFIED | int8: 3.2× smaller, +0.01 % perplexity |
| Speculative decoding | VERIFIED | output = greedy for 3 drafts; 1.16× at k = 2 |
| OpenAI-style HTTP server with streaming, disconnect handling | VERIFIED | `tests/server.rs` |
| Pretrained open-weights model (e.g. Llama) | VERIFIED (SmolLM2-135M-Instruct, CPU) | fetched from Docker Hub `ai/smollm2` with pinned sha256 (`tools/fetch_model.sh`); logits within 4.3e-5 of a float64 NumPy reference; 32-token greedy output equal to NumPy and llama.cpp; perplexity 18.8685 vs llama.cpp 18.8695 on the same 8,192 tokens ([pretrained.md](pretrained.md), `docs/evidence/2026-10-14/validate_smollm2_f16.json`) |
| GGUF reader (metadata, F32/F16/BF16/Q8_0 tensors) | VERIFIED | `tests/gguf.rs` (fixtures written by the `gguf` package; every metadata type; truncation and corruption); real F16 and Q8_0 files load |
| GPT-2 byte-level BPE tokenizer from GGUF, special tokens, chat template | VERIFIED | ids equal to Hugging Face `tokenizers` on 28/28 texts (14k tokens) and to llama.cpp on 27/28 (difference: control bytes with no token, documented); ChatML rendering equal to jinja2 |
| Qwen2 architecture (Q/K/V bias, NeoX RoPE) | PARTIAL | tiny synthetic Qwen2 GGUF matches NumPy in CI; the real Qwen2.5-0.5B was not run |
| `/v1/chat/completions` | IMPLEMENTED BUT UNVERIFIED | endpoint added; the chat template path is tested through `llm chat` and the opt-in test, not through HTTP |
| Benchmarks on the pretrained model (single stream, batching, TTFT, prefix cache, int8, vs llama.cpp) | VERIFIED (noisy) | `bench/results/2026-10-15/`; shared VM with load average 3–25, so speeds are rough; the llama.cpp speed comparison uses llama-cpp-python and is a lower bound for llama.cpp ([benchmarks.md](benchmarks.md)) |
| GPU execution | BLOCKED / NOT IMPLEMENTED | no GPU here, and the engine has no GPU path (CPU-only; see [OWNER_ACTIONS.md](OWNER_ACTIONS.md)) |
| CI | VERIFIED | `.github/workflows/llm-inference-engine.yml` (run 38560348688 green) |
