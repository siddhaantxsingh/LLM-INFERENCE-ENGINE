# Design

## The model

This section is about the small model trained here. The engine also runs pretrained GGUF models; see [pretrained.md](pretrained.md).

No pretrained weights could be downloaded here: huggingface.co and the PyTorch index are unreachable. So the model was trained for this project:

* **Architecture.** A Llama-architecture decoder (`train/model.py`, JAX on CPU): RMSNorm, rotary position embeddings, grouped-query attention (8 query heads, 2 KV heads), a SwiGLU MLP, and tied input/output embeddings.
* **Size.** 4 layers, d = 256, 3.0 M parameters, context 512.
* **Data.** The Python 3.12 standard library source (10 MB). Files are split train / held-out by path hash.
* **Tokenizer.** A byte-level BPE (`train/tokenizer.py`) with 1,024 tokens (768 merges), trained on the same corpus: 2.33 bytes per token. Merges never cross pre-tokenizer chunks (words, space runs, newline runs, single punctuation bytes).
* **Draft model.** A 1-layer, d = 128 draft model, trained the same way, for speculative decoding.

`train/export.py` writes the weights as `LLMW`: a JSON header (config, tensor names, shapes and offsets) followed by 64-byte-aligned little-endian f32 tensors.

## Engine (`src/`)

| Module | Role |
|---|---|
| `gguf.rs` | GGUF reader and dequantisation (F32, F16, BF16, Q8_0) |
| `tokenizer.rs` | the same BPE as `train/tokenizer.py` (tested against its output on held-out text), and GPT-2 style byte-level BPE from GGUF with special tokens and ChatML templates |
| `ops.rs` | AVX2/FMA dot products and matmuls (f32 and int8 weights) with a portable fallback |
| `model.rs` | weight loading (LLMW or GGUF `llama`/`qwen2`); the batched forward pass over a flat list of tokens from many sequences |
| `kv.rs` | paged KV cache and block pool with reference counts and prefix caching |
| `engine.rs` | scheduler: continuous batching, chunked prefill, preemption, sampling, stop strings, streaming text |
| `sampler.rs` | greedy, temperature, top-k, top-p, per-request seeded RNG |
| `spec.rs` | greedy speculative decoding with a draft model |
| `server.rs` | OpenAI-style `/v1/completions` (JSON or server-sent events), `/metrics` |

### Batch-invariant kernels

* Every dot product sums in one fixed order: four 8-lane fused multiply-add accumulators, a fixed reduction tree, then the tail. That order is the same in the AVX2 path and the portable path; a test checks they are bit-identical.
* A matmul computes each output from its own input row and weight row only.
* So a token's logits do not depend on what else is in the batch, on how a prompt is chunked, or on the thread count.
* The tests use this to demand *exact* equality between execution strategies, and speculative decoding relies on it.

### Forward pass

* A step's tokens can come from any number of sequences: decode tokens, prefill chunks, or both. They are stacked into one matrix, so every projection is one matmul in which each weight row is read once for the whole step. This is what makes batching pay on a memory-bound CPU.
* Per layer, each token's K and V are first written to its slot in the paged cache. Then each (token, head) pair attends over positions 0..=pos of its own sequence through that sequence's block table, in parallel.
* Logits are computed only for the last token of each sequence in the step.

### Paged KV cache and prefix caching

* The cache is a pool of blocks of 16 positions, covering all layers. A sequence holds a block table.
* **Registering blocks.** A block that fills up is registered under a *chain hash* of every token from position 0 to its end. Equal content at a different position, or after a different prefix, therefore gets a different hash.
* **Reusing blocks.** On admission, the prompt's leading full blocks are looked up in order, and every hit is reused with a reference count. The block holding the last prompt token is always recomputed, so a shared block is never written.
* **Releasing blocks.** A block that drops to zero references stays cached, and is evicted least-recently-released first, only when space is needed.

### Scheduler

Each `step()` builds one batch under a token budget (`max_batch_tokens`) and a sequence limit (`max_seqs`):

1. Decoding sequences contribute one token each.
2. Sequences still prefilling contribute a chunk of their prompt.
3. Waiting requests are admitted (prefix cache first) while there is budget, room and KV space.

**Preemption.** When KV blocks run out, the newest unscheduled sequence is preempted. It releases its blocks and returns to the front of the queue with everything it has generated, to be recomputed later. Recomputation is exact, so outputs never change; the tests check this, with freed blocks poisoned with NaN.

**Output.**

* Text is emitted incrementally: only complete UTF-8, and holding back a possible prefix of a stop string.
* A request finishes on `max_tokens`, on a stop string (cut before it), or at the context limit.

### Speculative decoding (`spec.rs`)

1. The draft model proposes k tokens.
2. The target scores the last accepted token plus the k proposals in **one** forward pass.
3. The longest agreeing prefix is accepted, plus the target's own token at the first disagreement.

Batch invariance makes the result identical to the target's plain greedy decoding. This is tested for drafts that always agree, mostly agree and often disagree.

### Server

* The engine runs on a dedicated thread. Each loop it drains new requests, runs one step, and routes events to per-request channels.
* HTTP handlers stream them as server-sent events, or collect them into one JSON reply.
* A client that disconnects closes its channel. The engine then aborts the sequence and frees its blocks.
