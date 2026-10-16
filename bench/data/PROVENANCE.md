# Text provenance

`alice_ch1-3.txt` is chapters I to III of *Alice's Adventures in Wonderland* by Lewis Carroll (1865).
The work is in the public domain.

* Source: Project Gutenberg eBook #11, through the GITenberg mirror on GitHub
  (gutenberg.org itself is unreachable from the build machine):
  `https://raw.githubusercontent.com/GITenberg/Alice-s-Adventures-in-Wonderland_11/master/11.txt`
* sha256 of the downloaded file: `0f9ea0b148d553177962a25edd2f56d36342c22576a3253a127b4fbeffa5687d` (fetched 2026-10-14).
* Extraction: lines 41 to 697 (from "CHAPTER I." up to, not including, "CHAPTER IV."), CRLF converted to LF.
  `sed -n '41,697p' 11.txt | tr -d '\r' > alice_ch1-3.txt`
* sha256 of `alice_ch1-3.txt`: `06e7f9ebe93f0865c6d155f0f07be8959b8a67559708f5ff27d11dd89d25b70e`.
* The Project Gutenberg header and licence text are not included, so the file carries no Project Gutenberg trademark.

This text is very likely in the pretraining data of SmolLM2 and Qwen2.5 (it is widely copied on the web).
Its perplexity is a fixed, reproducible measurement for comparing implementations, not a held-out quality score.
