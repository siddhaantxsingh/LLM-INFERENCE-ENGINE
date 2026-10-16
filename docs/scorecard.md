# Scorecard (2026-10-15)

Scores are 0–10, judged against what a hiring reviewer or a production team would expect. Evidence: [pretrained.md](pretrained.md), [benchmarks.md](benchmarks.md), [testing.md](testing.md).

| dimension | score | why not higher |
|---|---|---|
| Fresher-portfolio strength | 8 | A real pretrained model now runs and is cross-checked against NumPy and llama.cpp; but it is a 135M model, and the engine is slower than llama.cpp would be in a fair benchmark. |
| Engineering maturity | 7 | 28 CI tests with independent-reference fixtures, fuzzed GGUF reader, exactness tests; but no k-quant support, no HTTP test for chat completions, and benchmarks only on a shared, noisy VM. |
| Deployment readiness | 4 | One-binary server with streaming and a fetch script; no auth, TLS, rate limits, model hot-swap, GPU path or production-scale load testing. |
| Completion of intended scope | 7 | SmolLM2 load, tokenizer, validation, perplexity, benchmarks and tests are done; Qwen2.5-0.5B was not run (only a synthetic Qwen2 file), GPU is blocked, and the directive's C++/CUDA/Python stack was not used (Rust kept, see pretrained.md). |
