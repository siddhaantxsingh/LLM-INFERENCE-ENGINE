# Testing

`cargo test --release` runs 28 tests (plus one opt-in test that needs the real model) in about 30 s.

## Against an independent reference

* **`logits_match_the_numpy_reference`.** For three prompts, the engine's logits are compared with `train/reference.py`, a separate float64 NumPy forward pass of the exported weights. The maximum absolute difference is under 2·10⁻⁴. The KV blocks are deliberately non-contiguous and reversed.
* **`greedy_generation_matches_the_reference`.** 24 greedy tokens per prompt equal the reference, which recomputes the whole sequence at every step.
* **`tokenizer_matches_python`.** The Rust BPE encodes three held-out source files to exactly the token ids of `train/tokenizer.py`, and decoding round-trips.
* **Held-out loss.** `llm ppl` and `train/eval_ref.py` (JAX) evaluate the same windows. They agree to about 7 significant digits ([benchmarks.md](benchmarks.md)).

## Execution strategies must not change a single token

The kernels are batch-invariant, so these tests demand **exact** equality with a reference run: one request at a time, the whole prompt in one step, no cache. Freed KV blocks are overwritten with NaN, and so is the whole cache at start, so any read of stale or unwritten cache data corrupts the output.

* **`batching_chunking_and_caching_do_not_change_outputs`.** 8 requests, mixing greedy and seeded top-k/top-p sampling, under:
  * full batching;
  * prefill in 7-token chunks;
  * 4-token KV blocks;
  * at most 2 sequences at a time.
* **`preemption_recomputes_without_changing_outputs`.** A 12-block pool forces preemptions; the outputs are unchanged and every block is returned.
* **`prefix_cache_reuses_shared_prompts`.** A shared prompt is served from the cache (cached tokens are counted) with identical outputs.
* **`prefix_cache_keys_include_the_whole_prefix`.** A block with the same tokens after a different prefix, or at a different position, is not reused.
* **`speculative_decoding_equals_greedy`.** Three drafts, each at k = 1, 3 and 6, give exactly the target's greedy output:
  * the target itself (everything accepted);
  * its int8 copy (mostly accepted);
  * the trained 1-layer draft model (often rejected).

## Other behaviour

* **`stop_strings_and_limits`.** Streamed text equals the decoded tokens. Stop strings cut before the match. The context limit truncates generation. Empty or out-of-vocabulary prompts are rejected.
* **`int8_weights_stay_close_to_f32`.** int8 weights are smaller, and top-1 agreement with f32 is at least 90 %.
* **`seeds_control_sampling`.** The same seed gives the same output; different seeds diverge.
* **Unit tests.**
  * The SIMD and portable dot products are bit-identical.
  * A matmul is batch-invariant.
  * The sampler's empirical frequencies match its distribution within 0.005.
  * top-k/top-p supports are correct.
  * LRU eviction of cached blocks works.
* **Server (`tests/server.rs`).**
  * 4 concurrent HTTP requests return the same text as offline generation.
  * The SSE stream concatenates to the same text and ends in `[DONE]`.
  * Stop strings and parameter validation work.
  * A client that disconnects mid-stream has its sequence aborted and its KV blocks freed (checked through `/metrics`).

## Mutation checks

Each of these deliberate bugs makes at least one test fail with the trained model:

* attention reads one position too few;
* the RoPE sign is flipped;
* the prefix hash ignores its parent;
* the prefix match over-counts by one block;
* sampling ignores the per-request RNG;
* speculative verification is misaligned by one position.

One mutant survives: not resetting a preempted sequence's progress counter. A preempted sequence's first readmission attempt fails for lack of blocks in the same step, and that path resets the counter anyway. The bug is only reachable if readmission succeeds immediately, which the tests do not construct.

Three mutants survived the first version of the suite:

* the preemption one above;
* the prefix hash ignoring its parent: caught after adding the different-prefix and different-position cases;
* sampling ignoring the per-request RNG: caught after adding the seed test.

## Pretrained models (GGUF)

`tests/gguf.rs` (8 tests in CI, 1 opt-in). Details and the full-model validation results are in [pretrained.md](pretrained.md).

* **`tiny_llama_matches_numpy_reference`, `tiny_qwen2_with_biases_and_neox_rope_matches_numpy_reference`.** Tiny GGUF files written by the `gguf` Python package (random weights in F32/F16/BF16/Q8_0). Token ids, the argmax at every position and the last-position logits (within 2e-4 relative) match the float64 NumPy reference; 16 greedy tokens through the batched, paged engine (4-token blocks, NaN-poisoned) equal the reference's.
* **`tokenizer_matches_hf_tokenizers_and_llama_cpp`.** 25 varied texts per tokenizer: ids equal Hugging Face `tokenizers` and llama.cpp; decoding round-trips.
* **`chat_template_matches_jinja`.** ChatML rendering equals jinja2 for 3 conversations per template (SmolLM2's real template, and a Qwen-style template whose default-system literal must be told apart from the one that continues with the user's system message; the first version of the code failed this).
* **`end_of_sequence_token_stops_generation`.** The EOS token ends a request and adds no text; `ignore_eos` keeps going.
* **`reader_parses_every_metadata_and_tensor_type`, `reader_rejects_corrupt_files_without_panicking`, `unsupported_files_give_clear_errors`.** A GGUF built in the test with all 13 metadata types, nested arrays, 64-byte alignment and four tensor types; every truncated prefix is rejected; bad magic/version and absurd lengths fail fast.
* **Opt-in `smollm2_tokenizer_and_chat`** (`LLM_GGUF_MODEL=... cargo test --release --test gguf -- --ignored`): real-model token ids equal llama.cpp's, the chat prompt equals jinja2's rendering, and the answer to "What is the capital of France?" contains "Paris" and ends at `<|im_end|>`. Passed here: `docs/evidence/2026-10-14/optin_smollm2_test.log`.
* **`validate/compare.py`** (not part of `cargo test`; needs the model, numpy, gguf, tokenizers and llama-cpp-python): the full-model logits, tokenizer, greedy and perplexity comparisons.
