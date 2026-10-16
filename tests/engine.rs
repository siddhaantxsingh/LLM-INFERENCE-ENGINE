//! Engine correctness against the independent NumPy reference and between
//! its own execution strategies (batching, chunking, prefix caching,
//! preemption must not change a single token).

use llm_engine::engine::{Engine, EngineConfig, FinishReason};
use llm_engine::model::{Model, StepToken};
use llm_engine::sampler::SamplingParams;
use llm_engine::tokenizer::Tokenizer;
use serde_json::Value;
use std::sync::{Arc, OnceLock};

fn root() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn model() -> Arc<Model> {
    static M: OnceLock<Arc<Model>> = OnceLock::new();
    M.get_or_init(|| Arc::new(Model::load(&root().join("models/small.llmw")).unwrap()))
        .clone()
}

fn qmodel() -> Arc<Model> {
    static M: OnceLock<Arc<Model>> = OnceLock::new();
    M.get_or_init(|| {
        let mut m = Model::load(&root().join("models/small.llmw")).unwrap();
        m.quantize();
        Arc::new(m)
    })
    .clone()
}

fn tok() -> Arc<Tokenizer> {
    static T: OnceLock<Arc<Tokenizer>> = OnceLock::new();
    T.get_or_init(|| Arc::new(Tokenizer::from_json(&std::fs::read_to_string(root().join("models/tokenizer.json")).unwrap()).unwrap()))
        .clone()
}

fn fixtures() -> Vec<Value> {
    serde_json::from_str(&std::fs::read_to_string(root().join("tests/fixtures/reference.json")).unwrap()).unwrap()
}

fn engine(cfg: EngineConfig) -> Engine {
    // freed KV blocks become NaN: reading stale cache data cannot go unnoticed
    Engine::new(model(), tok(), EngineConfig { poison_freed: true, ..cfg })
}

fn prompts() -> Vec<Vec<u32>> {
    let t = tok();
    [
        "def fibonacci(n):\n    ",
        "import os\nimport sys\n\n",
        "class Stack:\n    def __init__(self):\n        self.",
        "    for i in range(len(",
        "# Copyright (C) 2001 Python Software Foundation\n\"\"\"",
        "        if not isinstance(value, str):\n            raise TypeError(",
        "x",
        "def _check(self, arg, kind):\n    \"\"\"Return the argument.\"\"\"\n    if arg is None:\n        return ",
    ]
    .iter()
    .map(|p| t.encode(p.as_bytes()))
    .collect()
}

fn greedy(n: usize) -> SamplingParams {
    SamplingParams {
        max_tokens: n,
        ..Default::default()
    }
}

#[test]
fn tokenizer_matches_python() {
    let f: Value = serde_json::from_str(&std::fs::read_to_string(root().join("tests/fixtures/tokenizer_fixture.json")).unwrap()).unwrap();
    let text: Vec<u8> = (0..f["text_hex"].as_str().unwrap().len() / 2)
        .map(|i| u8::from_str_radix(&f["text_hex"].as_str().unwrap()[2 * i..2 * i + 2], 16).unwrap())
        .collect();
    let want: Vec<u32> = serde_json::from_value(f["ids"].clone()).unwrap();
    let got = tok().encode(&text);
    assert_eq!(got.len(), want.len());
    assert_eq!(got, want);
    assert_eq!(tok().decode(&got), text);
}

#[test]
fn logits_match_the_numpy_reference() {
    let m = model();
    let bs = 16;
    for f in fixtures() {
        let ids: Vec<u32> = serde_json::from_value(f["tokens"].clone()).unwrap();
        let want: Vec<f64> = serde_json::from_value(f["logits_last"].clone()).unwrap();
        let mut kv = llm_engine::kv::KvCache::new(m.cfg.layers, m.cfg.kv_heads, m.cfg.head_dim(), 64, bs);
        let blocks: Vec<u32> = (0..64).rev().collect(); // deliberately non-contiguous
        let toks: Vec<StepToken> = ids
            .iter()
            .enumerate()
            .map(|(p, &t)| StepToken {
                token: t,
                pos: p,
                blocks: &blocks,
                want_logits: p + 1 == ids.len(),
            })
            .collect();
        let got = m.forward(&toks, &mut kv);
        let err = got[0].iter().zip(&want).map(|(a, b)| (*a as f64 - b).abs()).fold(0.0, f64::max);
        assert!(err < 2e-4, "max abs logit error {err}");
    }
}

#[test]
fn greedy_generation_matches_the_reference() {
    let mut e = engine(EngineConfig::default());
    for f in fixtures() {
        let ids: Vec<u32> = serde_json::from_value(f["tokens"].clone()).unwrap();
        let want: Vec<u32> = serde_json::from_value(f["greedy"].clone()).unwrap();
        let out = e.generate(vec![(ids, greedy(want.len()))]);
        assert_eq!(out[0].0, want, "prompt {:?}", f["prompt"]);
    }
}

fn run_all(e: &mut Engine, params: &[SamplingParams]) -> Vec<Vec<u32>> {
    let reqs = prompts().into_iter().zip(params.iter().cloned()).collect();
    let out = e.generate(reqs);
    e.pool_check().unwrap();
    out.into_iter().map(|o| o.0).collect()
}

fn mixed_params() -> Vec<SamplingParams> {
    (0..8)
        .map(|i| {
            if i % 2 == 0 {
                greedy(20 + i)
            } else {
                SamplingParams {
                    temperature: 0.8,
                    top_p: 0.95,
                    top_k: 50,
                    seed: i as u64,
                    max_tokens: 30,
                    stop: vec![],
                    ignore_eos: false,
                }
            }
        })
        .collect()
}

#[test]
fn batching_chunking_and_caching_do_not_change_outputs() {
    let params = mixed_params();
    // reference: one request at a time, whole prompt in one step, no cache
    let solo: Vec<Vec<u32>> = prompts()
        .into_iter()
        .zip(&params)
        .map(|(p, sp)| {
            let mut e = engine(EngineConfig {
                max_seqs: 1,
                prefix_caching: false,
                ..Default::default()
            });
            e.generate(vec![(p, sp.clone())]).remove(0).0
        })
        .collect();
    for (name, cfg) in [
        (
            "batched",
            EngineConfig {
                prefix_caching: false,
                ..Default::default()
            },
        ),
        (
            "chunked prefill (7 tokens/step)",
            EngineConfig {
                max_batch_tokens: 7,
                ..Default::default()
            },
        ),
        (
            "block size 4",
            EngineConfig {
                block_size: 4,
                ..Default::default()
            },
        ),
        (
            "2 sequences at a time",
            EngineConfig {
                max_seqs: 2,
                ..Default::default()
            },
        ),
    ] {
        let mut e = engine(cfg);
        assert_eq!(run_all(&mut e, &params), solo, "{name}");
    }
}

#[test]
fn preemption_recomputes_without_changing_outputs() {
    let params = mixed_params();
    let mut big = engine(EngineConfig {
        prefix_caching: false,
        ..Default::default()
    });
    let want = run_all(&mut big, &params);
    // room for about two sequences: the scheduler must preempt
    let mut small = engine(EngineConfig {
        num_blocks: 12,
        block_size: 8,
        prefix_caching: false,
        ..Default::default()
    });
    let got = run_all(&mut small, &params);
    assert!(small.stats.preemptions > 0, "expected preemptions");
    assert_eq!(got, want);
    assert_eq!(small.stats.pool.in_use, 0, "all blocks returned");
}

#[test]
fn prefix_cache_reuses_shared_prompts() {
    let t = tok();
    let system = "\"\"\"Utility functions for parsing configuration files and validating their contents.\"\"\"\nimport re\nimport os\n\n";
    let reqs: Vec<(Vec<u32>, SamplingParams)> = ["def load(path):\n", "def save(path, data):\n", "class Config:\n"]
        .iter()
        .map(|s| (t.encode(format!("{system}{s}").as_bytes()), greedy(12)))
        .collect();
    let mut nocache = engine(EngineConfig {
        prefix_caching: false,
        ..Default::default()
    });
    let want = nocache.generate(reqs.clone());
    let mut e = engine(EngineConfig::default());
    let first = e.generate(reqs[..1].to_vec());
    let rest = e.generate(reqs[1..].to_vec());
    assert_eq!(first[0].0, want[0].0);
    for (g, w) in rest.iter().zip(&want[1..]) {
        assert_eq!(g.0, w.0);
        assert!(g.2.cached_tokens >= 16, "cached {} tokens", g.2.cached_tokens);
    }
    // the identical prompt again: everything but the last block comes from cache
    let again = e.generate(reqs[..1].to_vec());
    assert_eq!(again[0].0, want[0].0);
    assert!(again[0].2.cached_tokens + 16 >= reqs[0].0.len());
    e.pool_check().unwrap();
}

#[test]
fn stop_strings_and_limits() {
    let mut e = engine(EngineConfig::default());
    let p = tok().encode(b"def fibonacci(n):\n    ");
    let full = e.generate(vec![(p.clone(), greedy(40))]).remove(0);
    assert_eq!(full.0.len(), 40);
    assert_eq!(full.1.as_bytes(), &tok().decode(&full.0)[..], "streamed text equals decoded tokens");
    // stop at the first newline the model produces, if any
    if let Some(pos) = full.1.find('\n') {
        let mut sp = greedy(40);
        sp.stop = vec!["\n".into()];
        let (_, text, usage) = e.generate(vec![(p.clone(), sp)]).remove(0);
        assert_eq!(text, full.1[..pos]);
        assert!(usage.completion_tokens <= 40);
    }
    // context limit
    let long: Vec<u32> = std::iter::repeat_n(p.clone(), 200)
        .flatten()
        .take(model().cfg.max_seq - 3)
        .collect();
    let (t, _, _) = e.generate(vec![(long, greedy(100))]).remove(0);
    assert_eq!(t.len(), 3);
    assert!(e.add(1, vec![], greedy(1)).is_err());
    assert!(e.add(2, vec![5000], greedy(1)).is_err());
    let _ = FinishReason::Stop;
}

#[test]
fn int8_weights_stay_close_to_f32() {
    let m = model();
    let q = qmodel();
    assert!(q.weight_bytes() * 3 < m.weight_bytes() * 2, "int8 matrices are smaller");
    let mut agree = 0;
    let mut total = 0;
    for f in fixtures() {
        let ids: Vec<u32> = serde_json::from_value(f["tokens"].clone()).unwrap();
        let bs = 16;
        let blocks: Vec<u32> = (0..64).collect();
        let toks: Vec<StepToken> = ids
            .iter()
            .enumerate()
            .map(|(p, &t)| StepToken {
                token: t,
                pos: p,
                blocks: &blocks,
                want_logits: true,
            })
            .collect();
        let mut kv1 = llm_engine::kv::KvCache::new(m.cfg.layers, m.cfg.kv_heads, m.cfg.head_dim(), 64, bs);
        let mut kv2 = llm_engine::kv::KvCache::new(m.cfg.layers, m.cfg.kv_heads, m.cfg.head_dim(), 64, bs);
        let a = m.forward(&toks, &mut kv1);
        let b = q.forward(&toks, &mut kv2);
        for (x, y) in a.iter().zip(&b) {
            total += 1;
            agree += (llm_engine::sampler::argmax(x) == llm_engine::sampler::argmax(y)) as usize;
            let err = x.iter().zip(y).map(|(p, q)| (p - q).abs()).fold(0f32, f32::max);
            assert!(err < 0.5, "int8 logit error {err}");
        }
    }
    assert!(agree * 10 >= total * 9, "top-1 agreement {agree}/{total}");
}

#[test]
fn prefix_cache_keys_include_the_whole_prefix() {
    // block 1 has the same 16 tokens in both prompts, after different block 0s
    let a: Vec<u32> = (0..16).map(|i| 300 + i).chain((0..16).map(|i| 500 + i)).chain([7, 8]).collect();
    let b: Vec<u32> = (0..16).map(|i| 600 + i).chain((0..16).map(|i| 500 + i)).chain([7, 8]).collect();
    let mut nocache = engine(EngineConfig {
        prefix_caching: false,
        ..Default::default()
    });
    let want = nocache.generate(vec![(b.clone(), greedy(10))]);
    let mut e = engine(EngineConfig::default());
    e.generate(vec![(a.clone(), greedy(10))]);
    let got = e.generate(vec![(b, greedy(10))]);
    assert_eq!(got[0].2.cached_tokens, 0, "a block after a different prefix must not be reused");
    assert_eq!(got[0].0, want[0].0);
    // the same 16 tokens at position 0 instead of 16: different RoPE positions, no reuse
    let c: Vec<u32> = a[16..].to_vec();
    let want = nocache.generate(vec![(c.clone(), greedy(10))]);
    let got = e.generate(vec![(c, greedy(10))]);
    assert_eq!(got[0].2.cached_tokens, 0, "a block at a different position must not be reused");
    assert_eq!(got[0].0, want[0].0);
}

#[test]
fn seeds_control_sampling() {
    let p = tok().encode(b"def ");
    let sp = |seed| SamplingParams {
        temperature: 1.0,
        seed,
        max_tokens: 24,
        ..Default::default()
    };
    let mut e = engine(EngineConfig::default());
    let out = e.generate(vec![(p.clone(), sp(1)), (p.clone(), sp(1)), (p.clone(), sp(2)), (p.clone(), sp(3))]);
    assert_eq!(out[0].0, out[1].0, "same seed, same output");
    assert!(out[0].0 != out[2].0 || out[0].0 != out[3].0, "different seeds diverge");
}

#[test]
fn speculative_decoding_equals_greedy() {
    let target = model();
    let mut e = engine(EngineConfig::default());
    // drafts: the target itself (everything accepted), its int8 copy (mostly),
    // and the trained draft model when present
    let mut drafts: Vec<(&str, Arc<Model>)> = vec![("self", model()), ("int8", qmodel())];
    let dp = root().join("models/draft.llmw");
    if dp.exists() {
        drafts.push(("draft", Arc::new(Model::load(&dp).unwrap())));
    }
    for p in prompts() {
        let want = e.generate(vec![(p.clone(), greedy(40))]).remove(0).0;
        for (name, d) in &drafts {
            for k in [1, 3, 6] {
                let (got, st) = llm_engine::spec::generate(&target, d, &p, 40, k);
                assert_eq!(got, want, "draft {name}, k={k}");
                assert!(st.target_passes <= 41);
                if *name == "self" {
                    assert_eq!(st.accepted, st.proposed, "a model always agrees with itself");
                }
            }
        }
    }
}
