# LLM inference engine (project 7)

A CPU inference engine for Llama-architecture models, written in Rust. It runs a real pretrained model, **SmolLM2-135M-Instruct** (GGUF), and a small model trained here. It provides:

* a paged KV cache with **prefix caching**;
* **continuous batching** with chunked prefill and preemption;
* int8 weight quantisation and **speculative decoding**;
* seeded sampling (temperature, top-k, top-p, stop strings);
* an OpenAI-style HTTP server with server-sent-event streaming (`/v1/completions`, `/v1/chat/completions`);
* a GGUF reader (F32/F16/BF16/Q8_0), Llama and Qwen2 architectures, and a GPT-2 style byte-level BPE tokenizer with special tokens and ChatML chat templates.

huggingface.co is blocked here, so `tools/fetch_model.sh` fetches SmolLM2 from Docker Hub's `ai/` namespace (pinned sha256, licence recorded). The original small model (3.0 M parameters, trained with JAX in `train/` on Python source) is still the default for the fast tests.

**Status: CPU, verified.** All tests pass. SmolLM2's logits match an independent float64 NumPy forward pass to 4e-5 and its greedy output equals llama.cpp's. The engine has no GPU path (no GPU here; the CUDA work is in `cuda-kernels/`). Rust was kept although the owner's directive named C++/CUDA/Python; the reason is in [docs/pretrained.md](docs/pretrained.md#why-rust-not-ccudapython).

## Highlights

From [docs/pretrained.md](docs/pretrained.md), [docs/benchmarks.md](docs/benchmarks.md) and [docs/testing.md](docs/testing.md):

* **SmolLM2-135M-Instruct checked against two independent references.** Logits on 5 prompts are within 4.3·10⁻⁵ (relative 1.3·10⁻⁶) of a float64 NumPy implementation; top-1 agrees with llama.cpp at every position; 32 greedy tokens are identical to both. Perplexity on 8,192 tokens of *Alice in Wonderland*: 18.8685 (engine) vs 18.8695 (llama.cpp).
* **Tokenizer** ids equal Hugging Face `tokenizers` on 28/28 varied texts (14k tokens; unicode, code, whitespace, special tokens) and llama.cpp on 27/28 (the exception is explained).

For the small trained model:

* **Matches the trainer.** The engine's held-out loss equals the JAX trainer's to 1.8·10⁻⁸ nats/token over 59k tokens. Logits match an independent float64 NumPy forward pass to under 2·10⁻⁴.
* **Batch-invariant kernels.** Every dot product sums in one fixed order, in both the AVX2 and the portable path. So batching, chunked prefill, prefix caching, preemption and speculative decoding give **bit-identical** outputs to one-at-a-time decoding, and the tests require exactly that. Freed KV blocks are poisoned with NaN, so stale cache reads cannot go unnoticed.
* **Continuous batching** raises throughput 3.1× on 4 vCPUs (610 → 1,883 output tokens/s).
* **int8 weights** are 3.2× smaller, with +0.01 % perplexity.

## Quick start

```sh
cargo build --release
tools/fetch_model.sh smollm2-135m-instruct-f16     # 271 MB into /home/user/models/
./target/release/llm chat --model /home/user/models/smollm2-135m-instruct-f16.gguf --prompt 'What is the capital of France?'
./target/release/llm serve --model /home/user/models/smollm2-135m-instruct-f16.gguf --ctx 2048 --listen 127.0.0.1:8080
curl -s localhost:8080/v1/chat/completions -d '{"messages": [{"role": "user", "content": "Name three colours."}], "max_tokens": 40}'

./target/release/llm generate --prompt 'def fibonacci(n):' --max-tokens 60
./target/release/llm serve --listen 127.0.0.1:8080 [--q8] [--blocks 2048] [--max-seqs 64]
curl -s localhost:8080/v1/completions -d '{"prompt": "import os\n", "max_tokens": 32, "temperature": 0.8, "top_p": 0.95, "seed": 7}'
curl -sN localhost:8080/v1/completions -d '{"prompt": "class Stack:\n", "max_tokens": 32, "stream": true}'
./target/release/llm ppl --text FILE [--q8]           # held-out loss
./target/release/llm bench --text FILE                # throughput vs batch size
./target/release/llm spec --text FILE                 # speculative decoding vs greedy
```

Retrain from scratch, after `pip install jax`:

```sh
cd train
python3 build_tokenizer.py 1024
python3 train.py small 3000 artifacts/small.pkl
python3 export.py artifacts/small.pkl ../models/small.llmw
```

## Documentation

* [Pretrained models](docs/pretrained.md): fetching, GGUF, tokenizer, validation against NumPy and llama.cpp, why Rust.
* [Design](docs/design.md): the model, the forward pass, paged KV and prefix caching, the scheduler, speculative decoding and the server.
* [Testing](docs/testing.md): reference checks, exactness across execution strategies, and mutation checks.
* [Benchmarks](docs/benchmarks.md); reproduce them with `bench/run.sh`.
* [Limitations](docs/limitations.md).
* [Requirements matrix](docs/requirements.md).
* [Owner actions](docs/OWNER_ACTIONS.md).
* [Scorecard](docs/scorecard.md).
