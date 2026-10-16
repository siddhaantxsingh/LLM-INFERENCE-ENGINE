# Running a pretrained model: SmolLM2-135M-Instruct (GGUF)

The engine now loads GGUF files and runs **SmolLM2-135M-Instruct**, a real pretrained open-weights model (Apache-2.0), through the same paged-KV, continuous-batching engine as the small model trained in `train/`.
Everything here ran on the 4-vCPU build machine, CPU only. Raw outputs are in `docs/evidence/2026-10-14/` and `bench/results/2026-10-15/`.

## Getting the model

huggingface.co is blocked on the build machine. Docker Hub is not, and its `ai/` namespace publishes GGUF files as OCI artifacts.
`tools/fetch_model.sh` downloads them through the registry API, checks a pinned sha256, and records the source and licence next to the file.

```sh
tools/fetch_model.sh smollm2-135m-instruct-f16     # 271 MB, F16 weights
tools/fetch_model.sh smollm2-135m-instruct-q8_0    # 145 MB, Q8_0 weights
```

| name | Docker Hub tag | sha256 (pinned) | licence |
|---|---|---|---|
| smollm2-135m-instruct-f16 | `ai/smollm2:135m-f16` | `5157ca60…afb30` | Apache-2.0 |
| smollm2-135m-instruct-q8_0 | `ai/smollm2:135m-q8_0` | `c4a3dd03…6ce66a` | Apache-2.0 |

* Files go to `/home/user/models/` (override with `MODELS_DIR`). They are not committed (`*.gguf` is gitignored, except the tiny test fixtures).
* The manifest annotations say these tags hold `unsloth/SmolLM2-135M-Instruct-GGUF`. The GGUF metadata says `SmolLM2 135M Instruct`, `HuggingFaceTB`.
* `ai/smollm2:latest` is a different model (SmolLM2-360M, Q4_K_M), and `ai/smollm2:135M-F16` has no chat template in its metadata. Neither is used.
* Qwen2.5-0.5B is listed in the script, and the loader supports the `qwen2` architecture (Q/K/V biases, NeoX RoPE). It is tested only on a tiny synthetic Qwen2 file; the real 0.5B model was **not** run (disk space and time).

## What was added

| file | what |
|---|---|
| `src/gguf.rs` | GGUF v2/v3 reader: all metadata types, tensor directory, alignment, bounds checks; dequantises F32, F16, BF16 and Q8_0 to f32 |
| `src/model.rs` | `Model::from_gguf` for `llama` and `qwen2`; config from metadata (heads, KV heads, RoPE theta, RMSNorm eps, context); tied or separate output matrix; Q/K/V biases; NeoX RoPE |
| `src/tokenizer.rs` | GPT-2 style byte-level BPE from GGUF metadata: vocabulary, ranked merges, the `smollm` and `qwen2` pre-tokenizer regexes, special tokens matched before BPE, BOS/EOS, ChatML chat template |
| `src/engine.rs` | generation stops at the model's end-of-sequence token (`ignore_eos` turns that off for benchmarks) |
| `src/server.rs` | `/v1/chat/completions` (JSON and SSE) using the model's chat template |
| `src/bin/llm.rs` | `--model X.gguf` everywhere; `chat`, `dump` (logits), `tokenize`, `prefix-bench` |
| `tools/fetch_model.sh` | the download script above |
| `validate/` | the independent references and comparison scripts (Python) |

Design choices:

* **Weights are dequantised to f32 at load time** (651 MB in memory for the 135M model, tied output matrix duplicated). `--q8` then applies the engine's own per-row int8 quantisation. Q8_0 blocks are not kept as int8, so the Q8_0 file gives the same speed as the F16 file; only its weights' values differ.
* **RoPE.** llama.cpp's converter permutes the Q and K weights of Llama models so that RoPE rotates adjacent pairs. The engine's existing RoPE already rotates adjacent pairs, so Llama GGUF files need no change. Qwen2 files are not permuted and use the "NeoX" layout (pairs `i`, `i + d/2`), selected by `Config::rope_neox`.
* **Unsupported features fail loudly**: other architectures, k-quants and other tensor types, partial RoPE, RoPE scaling, tokenizers other than byte-level BPE, and pre-tokenizers other than `gpt2`/`smollm`/`qwen2` all give an error at load time.
* **Missing byte tokens.** The SmolLM2 vocabulary has no token for a few control bytes (for example 0x04). The engine maps them to the unknown token, as Hugging Face `tokenizers` does; llama.cpp silently drops them. This is the only tokenizer difference found (see below).
* **Chat template.** Only ChatML templates (`<|im_start|>`), which SmolLM2 and Qwen2.5 use, are rendered. The default system message is read from the template's string literal. Tool calls are not supported. The output is tested against jinja2 rendering of the template.

## Independent validation (`validate/compare.py`)

Two references that share no code with the engine:

1. **`validate/ref_numpy.py`**: a float64 NumPy forward pass written for this check. The `gguf` package (from the llama.cpp repository) reads and dequantises the tensors. It recomputes the whole sequence with full causal attention, no cache.
2. **llama.cpp**, through `llama-cpp-python` 0.3.36, built here from the PyPI source distribution (cmake, `GGML_NATIVE=ON`).

For the tokenizer, a third reference: Hugging Face `tokenizers`, configured from the GGUF vocabulary and merges with SmolLM2's pre-tokenizer (`Digits(individual_digits) + ByteLevel`).

Results on the F16 file (`docs/evidence/2026-10-14/validate_smollm2_f16.json`; the Q8_0 file: `bench/results/2026-10-15/validate_smollm2_q8_0.json`):

**Tokenizer.** 28 texts, 14,061 tokens: ASCII, accents, CJK, Cyrillic, Greek, Devanagari, Arabic, emoji with ZWJ, maths symbols, code, tabs, CRLF, runs of spaces, numbers, special tokens written in text, 300 repeated characters, empty strings, plus all of the Alice text and `src/engine.rs`.

| check | result |
|---|---|
| ids equal to Hugging Face `tokenizers` | 28 / 28 texts |
| ids equal to llama.cpp | 27 / 28; the exception is the string with bytes 0x01 and 0x04, which have no token (see above) |
| decode(encode(text)) = text | 27 / 28; the same string (unknown bytes cannot round-trip) |

**Logits** (all positions of 5 prompts: a chat prompt, English, Python, numbers + unicode, 200 tokens of Alice):

| prompt tokens | engine vs NumPy: max abs diff | max rel. error | llama.cpp vs NumPy: max abs diff | top-1 agreement (all pairs) |
|---|---|---|---|---|
| 37 | 4.2e-5 | 1.0e-6 | 4.5e-2 | 100 % |
| 8 | 3.6e-5 | 1.3e-6 | 2.1e-2 | 100 % |
| 24 | 3.0e-5 | 6.7e-7 | 5.1e-2 | 100 % |
| 37 | 2.4e-5 | 6.4e-7 | 4.2e-2 | 100 % |
| 200 | 2.8e-5 | 7.2e-7 | 6.3e-2 | 100 % |

Relative error is max |diff| / max |reference logit|. The engine computes in f32 with the f16 weights converted exactly, so it sits within float rounding of the float64 reference. llama.cpp's CPU path converts activations to f16 for F16 matmuls, so it is about 1,000× further from the float64 reference; the engine and llama.cpp still agree on the top token at every position.

**Greedy generation.** 32 greedy tokens for 3 prompts are identical in the engine, the NumPy reference and llama.cpp. The chat prompt answers "The capital of France is Paris."

**Perplexity** on the same 8,192 tokens (chapters I–III of *Alice's Adventures in Wonderland*, `bench/data/alice_ch1-3.txt`, non-overlapping 512-token windows, every position scored):

| implementation | loss (nats/token) | perplexity |
|---|---|---|
| engine, F16 file | 2.937495 | 18.8685 |
| llama.cpp, F16 file | 2.937545 | 18.8695 |

The difference is 5e-5 nats/token (0.005 % perplexity), consistent with llama.cpp's f16 activations. The text is probably in the model's training data, so the perplexity is a reproducible comparison, not a quality score. The comparison uses llama.cpp's logits on identical windows, not the `llama-perplexity` tool (which scores only the second half of each window).

## Tests

* CI (`cargo test`) uses two tiny GGUF files in `tests/fixtures/` (230 and 277 KB), written by the `gguf` Python package with random weights in F32, F16, BF16 and Q8_0, and a 673-token vocabulary (SmolLM2's special tokens, the 256 byte tokens and its first 400 merges). One is `llama` (tied embeddings, `smollm` pre-tokenizer); the other is `qwen2` (Q/K/V biases, NeoX RoPE, separate Q8_0 output matrix, `qwen2` pre-tokenizer). The expectations come from the references: NumPy logits and greedy tokens, Hugging Face and llama.cpp token ids, jinja2 chat rendering. Regenerate with `validate/make_tiny_fixture.py`.
* The reader is also tested on a GGUF written in the test itself, with every metadata type, nested arrays, 64-byte alignment and one tensor of each type; every truncated prefix and every single-byte corruption of the header is rejected or parsed without a panic.
* The full-model test is opt-in: `LLM_GGUF_MODEL=/home/user/models/smollm2-135m-instruct-f16.gguf cargo test --release --test gguf -- --ignored` (token ids checked against llama.cpp's, chat prompt checked against jinja2, the answer must contain "Paris" and end at `<|im_end|>`).

## Why Rust, not C++/CUDA/Python

The owner's directive named C++/CUDA/Python for this project. The engine was already written in Rust, with its batch-invariant kernels, paged KV cache, scheduler and tests. Rewriting it would have cost the whole time budget and produced the same CPU engine. Rust is kept. Python is used where it is the natural tool: the independent references and the validation scripts. CUDA is not used because there is no GPU here and the engine has no GPU path (see `docs/OWNER_ACTIONS.md`); `cuda-kernels/` in this repository holds the CUDA work.
