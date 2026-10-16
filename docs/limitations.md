# Limitations

* **Pretrained models: one family verified.**
  * SmolLM2-135M-Instruct (GGUF, F16 and Q8_0) is verified against NumPy and llama.cpp. Qwen2 support is tested only on a tiny synthetic file; Qwen2.5-0.5B was not run.
  * Only F32, F16, BF16 and Q8_0 tensors load; k-quants (Q4_K etc.) and i-quants are rejected. Weights are dequantised to f32 at load, so a Q8_0 file is no faster or smaller in memory than F16 (use `--q8` for the engine's own int8).
  * Only byte-level BPE tokenizers with the `gpt2`, `smollm` or `qwen2` pre-tokenizers, and only ChatML chat templates (no tool calls). No safetensors loader.
  * A byte with no vocabulary token becomes the unknown token (as in Hugging Face `tokenizers`); llama.cpp drops it instead.
  * The engine's single-stream decode is slow for this model (memory-bound f32 weights; see [benchmarks.md](benchmarks.md)). llama.cpp is faster.
* **The small model trained here** (3.0 M parameters on Python source) is still the default for tests; it is not useful as an assistant.
* **CPU only.** The kernels are AVX2+FMA with a portable fallback. There is no AVX-512/AMX path, no GPU, and no tensor parallelism.
* **Weight-only int8.** Activations stay f32. No 4-bit or group-wise quantisation, and no int8 KV cache.
* **Attention cost.** Attention is computed per (token, head) over the whole cached prefix. There is no FlashAttention-style tiling on CPU, so long contexts would be slow; the trained context is 512.
* **Preemption recomputes.** Preempted sequences are recomputed rather than swapped out. Recomputation is exact, but it repeats the work.
* **Speculative decoding** is greedy-only and single-sequence. It is not integrated into the batched scheduler; it is a separate API used by `llm spec`.
* **The server** has no authentication, rate limiting or TLS (see `inferx/` for a gateway). `/v1/chat/completions` exists (ChatML templates only) but is not covered by HTTP tests.
