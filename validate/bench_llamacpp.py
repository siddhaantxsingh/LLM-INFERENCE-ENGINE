"""llama.cpp (llama-cpp-python) single-stream speed on the same prompt shape
as `llm bench --requests 1`: prefill of P tokens, then G greedy tokens.

    python -I validate/bench_llamacpp.py MODEL.gguf TEXT P G THREADS RUNS
"""
import json
import sys
import time

import numpy as np
from llama_cpp import Llama

model, text, P, G, threads, runs = sys.argv[1], sys.argv[2], *map(int, sys.argv[3:7])
llm = Llama(model, n_ctx=1024, n_batch=512, n_threads=threads, n_threads_batch=threads, verbose=False)
ids = llm.tokenize(open(text, encoding="utf-8").read().encode(), add_bos=False)[:P]
for r in range(runs):
    llm.reset()
    t0 = time.perf_counter()
    llm.eval(ids)
    tok = int(np.argmax(llm.scores[llm.n_tokens - 1]))
    ttft = time.perf_counter() - t0
    t1 = time.perf_counter()
    for _ in range(G - 1):
        llm.eval([tok])
        tok = int(np.argmax(llm.scores[llm.n_tokens - 1]))
    dec = time.perf_counter() - t1
    print(json.dumps({"engine": "llama.cpp", "run": r, "threads": threads, "prompt_tokens": P, "gen_tokens": G,
                      "ttft_ms": ttft * 1e3, "decode_tokens_per_s": (G - 1) / dec}), flush=True)
