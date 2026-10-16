# Owner actions

Steps only the owner can take. Everything else in this project ran on the build machine.

## GPU (RTX 4060): nothing to run

The engine has **no GPU path**. Its kernels are AVX2/FMA with a portable fallback, and nothing in `src/` calls CUDA. So there is no GPU script to run for this project, and "GPU execution" stays BLOCKED / NOT IMPLEMENTED here. The CUDA work in this repository is in `cuda-kernels/`.

## Reproduce the pretrained-model results on your own machine (optional)

Needs Rust (1.89+), Python 3.10+, cmake and a C/C++ compiler.

```sh
cd llm-inference-engine
tools/fetch_model.sh smollm2-135m-instruct-f16      # prints: ...smollm2-135m-instruct-f16.gguf: ok (270885952 bytes, licence Apache-2.0)
tools/fetch_model.sh smollm2-135m-instruct-q8_0
python3 -m venv .venv && .venv/bin/pip install numpy gguf tokenizers jinja2 llama-cpp-python==0.3.36
cargo build --release
./target/release/llm chat --model /home/user/models/smollm2-135m-instruct-f16.gguf --prompt "What is the capital of France?"
#   expected: The capital of France is Paris.
LLM_GGUF_MODEL=/home/user/models/smollm2-135m-instruct-f16.gguf cargo test --release --test gguf -- --ignored
.venv/bin/python -I validate/compare.py --model /home/user/models/smollm2-135m-instruct-f16.gguf \
    --llm target/release/llm --out validate_f16.json --ppl bench/data/alice_ch1-3.txt
PY=.venv/bin/python bench/run_gguf.sh bench/results/$(date +%F)
```

Save `validate_f16.json` and the `bench/results/<date>/` folder. Expected: engine vs NumPy max |logit diff| below 1e-4, greedy tokens equal to llama.cpp, perplexity about 18.87 on the Alice text. Timings will differ from `docs/benchmarks.md` (a shared 4-vCPU VM).
