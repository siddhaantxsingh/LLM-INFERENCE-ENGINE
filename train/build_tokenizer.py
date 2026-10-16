import sys
import time

import data
import tokenizer

train, held = data.split()
corpus = data.read(train)
print(f"{len(train)} train files ({len(corpus) / 1e6:.1f} MB), {len(held)} held out")
t = time.time()
merges = tokenizer.train(corpus[: 3_000_000], int(sys.argv[1]) if len(sys.argv) > 1 else 1024)
tok = tokenizer.Tokenizer(merges)
print(f"trained {len(merges)} merges in {time.time() - t:.0f} s")
tok.save("artifacts/tokenizer.json")
ids = tok.encode(corpus)
assert tok.decode(ids) == corpus
print(f"{len(ids) / 1e6:.2f} M tokens, {len(corpus) / len(ids):.2f} bytes/token")
