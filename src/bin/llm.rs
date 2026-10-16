//! llm generate | chat | serve | ppl | bench | prefix-bench | spec | dump | tokenize   (see README for flags)

use llm_engine::engine::{Engine, EngineConfig};
use llm_engine::model::{Model, StepToken};
use llm_engine::sampler::SamplingParams;
use llm_engine::tokenizer::Tokenizer;
use serde_json::json;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

struct Args(HashMap<String, String>);

impl Args {
    fn parse(v: &[String]) -> Args {
        let mut m = HashMap::new();
        let mut i = 0;
        while i < v.len() {
            let k = v[i].trim_start_matches("--").to_string();
            if i + 1 < v.len() && !v[i + 1].starts_with("--") {
                m.insert(k, v[i + 1].clone());
                i += 2;
            } else {
                m.insert(k, "true".into());
                i += 1;
            }
        }
        Args(m)
    }
    fn get(&self, k: &str, d: &str) -> String {
        self.0.get(k).cloned().unwrap_or_else(|| d.to_string())
    }
    fn num<T: std::str::FromStr>(&self, k: &str, d: T) -> T
    where
        T::Err: std::fmt::Debug,
    {
        self.0.get(k).map(|v| v.parse().expect(k)).unwrap_or(d)
    }
    fn flag(&self, k: &str) -> bool {
        self.0.contains_key(k)
    }
}

/// The model (LLMW or GGUF) and its tokenizer. A GGUF file carries its own
/// tokenizer; an LLMW model uses --tokenizer.
fn load(a: &Args) -> (Arc<Model>, Arc<Tokenizer>) {
    let path = a.get("model", "models/small.llmw");
    let ctx = a.0.get("ctx").map(|v| v.parse::<usize>().expect("ctx"));
    let t0 = Instant::now();
    let (mut m, t) = Model::load_any(std::path::Path::new(&path), ctx).expect("load model");
    let t = t.unwrap_or_else(|| {
        Tokenizer::from_json(&std::fs::read_to_string(a.get("tokenizer", "models/tokenizer.json")).expect("tokenizer")).unwrap()
    });
    if a.flag("q8") {
        m.quantize();
    }
    eprintln!(
        "[loaded {path}: {} layers, d={}, vocab {}, context {}, {:.1} MB weights in memory, {:.2} s]",
        m.cfg.layers,
        m.cfg.dim,
        m.cfg.vocab,
        m.cfg.max_seq,
        m.weight_bytes() as f64 / 1e6,
        t0.elapsed().as_secs_f64()
    );
    (Arc::new(m), Arc::new(t))
}

/// Default KV pool: 2048 blocks, or fewer if that would exceed 1 GB.
fn default_blocks(m: &Model, bs: usize) -> usize {
    let per_block = 2 * m.cfg.layers * bs * m.cfg.kv_dim() * 4;
    2048.min((1usize << 30) / per_block).max(16)
}

fn engine_cfg(a: &Args, m: &Model) -> EngineConfig {
    let d = EngineConfig::default();
    let bs = a.num("block-size", d.block_size);
    EngineConfig {
        block_size: bs,
        num_blocks: a.num("blocks", default_blocks(m, bs)),
        max_batch_tokens: a.num("batch-tokens", d.max_batch_tokens),
        max_seqs: a.num("max-seqs", d.max_seqs),
        prefix_caching: !a.flag("no-prefix-cache"),
        poison_freed: false,
    }
}

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let cmd = argv.get(1).cloned().unwrap_or_default();
    let a = Args::parse(&argv[2.min(argv.len())..]);
    match cmd.as_str() {
        "generate" | "chat" => {
            let (m, t) = load(&a);
            let mut e = Engine::new(m.clone(), t.clone(), engine_cfg(&a, &m));
            let sp = SamplingParams {
                temperature: a.num("temperature", 0.0),
                top_p: a.num("top-p", 1.0),
                top_k: a.num("top-k", 0),
                seed: a.num("seed", 0),
                max_tokens: a.num("max-tokens", 128),
                stop: vec![],
                ignore_eos: a.flag("ignore-eos"),
            };
            let mut prompt = a.get("prompt", "def ");
            if cmd == "chat" {
                let mut msgs = vec![];
                if let Some(sys) = a.0.get("system") {
                    msgs.push(("system".to_string(), sys.clone()));
                }
                msgs.push(("user".to_string(), prompt));
                prompt = t.apply_chat_template(&msgs).expect("chat template");
            }
            let t0 = Instant::now();
            // --prompt-ids 1,2,3 gives the prompt as token ids (validation scripts)
            let ids = match a.0.get("prompt-ids") {
                Some(v) => v.split(',').map(|x| x.trim().parse::<u32>().expect("prompt-ids")).collect(),
                None => t.encode_prompt(&prompt),
            };
            let (toks, text, u) = e.generate(vec![(ids, sp)]).remove(0);
            if let Some(f) = a.0.get("ids-out") {
                std::fs::write(f, serde_json::to_string(&toks).unwrap()).expect("write --ids-out");
            }
            if cmd == "chat" {
                println!("{text}");
            } else {
                println!("{prompt}{text}");
            }
            eprintln!(
                "[{} tokens in {:.2} s, ttft {:.1} ms]",
                toks.len(),
                t0.elapsed().as_secs_f64(),
                u.ttft_ms
            );
        }
        "serve" => {
            tracing_subscriber::fmt().init();
            let (m, t) = load(&a);
            let cfg = engine_cfg(&a, &m);
            let e = Engine::new(m.clone(), t, cfg.clone());
            tracing::info!(
                ?cfg,
                kv_mb = e.kv_bytes() as f64 / 1e6,
                weights_mb = m.weight_bytes() as f64 / 1e6,
                "engine ready"
            );
            let base = std::path::Path::new(&a.get("model", "models/small.llmw"))
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default();
            let name = if a.flag("q8") { format!("{base}-q8") } else { base };
            let app = llm_engine::server::start(e, name);
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let l = tokio::net::TcpListener::bind(a.get("listen", "127.0.0.1:8080")).await.unwrap();
                tracing::info!("listening on {}", l.local_addr().unwrap());
                // stop on SIGINT or SIGTERM (process managers and containers send SIGTERM)
                let shutdown = async {
                    let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()).unwrap();
                    tokio::select! {
                        _ = tokio::signal::ctrl_c() => {}
                        _ = term.recv() => {}
                    }
                };
                axum::serve(l, app).with_graceful_shutdown(shutdown).await.unwrap();
            });
        }
        "ppl" => {
            // mean next-token loss over non-overlapping windows (as the trainer evaluates)
            let (m, t) = load(&a);
            let text = std::fs::read(a.get("text", "")).expect("--text FILE");
            let ids = t.encode(&text);
            let win: usize = a.num("window", 256);
            let mut kv = llm_engine::kv::KvCache::new(m.cfg.layers, m.cfg.kv_heads, m.cfg.head_dim(), win.div_ceil(16) + 1, 16);
            let blocks: Vec<u32> = (0..(win.div_ceil(16) + 1) as u32).collect();
            let (mut nll, mut n) = (0f64, 0usize);
            let t0 = Instant::now();
            // windows of win+1 tokens every win tokens (train/eval_ref.py uses the same)
            for w in (0..ids.len().saturating_sub(win)).step_by(win).map(|s| &ids[s..s + win + 1]) {
                let toks: Vec<StepToken> = w[..win]
                    .iter()
                    .enumerate()
                    .map(|(p, &tk)| StepToken {
                        token: tk,
                        pos: p,
                        blocks: &blocks,
                        want_logits: true,
                    })
                    .collect();
                let logits = m.forward(&toks, &mut kv);
                for (p, l) in logits.iter().enumerate() {
                    let mx = l.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)) as f64;
                    let z: f64 = l.iter().map(|&x| (x as f64 - mx).exp()).sum();
                    nll += mx + z.ln() - l[w[p + 1] as usize] as f64;
                    n += 1;
                }
            }
            let r = json!({"tokens": n, "loss": nll / n as f64, "perplexity": (nll / n as f64).exp(), "seconds": t0.elapsed().as_secs_f64(),
                           "quantized": a.flag("q8"), "weight_bytes": m.weight_bytes()});
            println!("{r}");
        }
        "bench" => bench(&a),
        "prefix-bench" => prefix_bench(&a),
        "dump" => dump(&a),
        "tokenize" => {
            // --input F: a JSON list of strings -> a JSON list of token-id lists (no BOS)
            let (_, t) = load(&a);
            let texts: Vec<String> = serde_json::from_str(&std::fs::read_to_string(a.get("input", "")).expect("--input FILE")).unwrap();
            let out: Vec<serde_json::Value> = texts
                .iter()
                .map(|x| {
                    let ids = t.encode(x.as_bytes());
                    json!({"ids": ids, "roundtrip": t.decode(&ids) == x.as_bytes()})
                })
                .collect();
            println!("{}", serde_json::to_string(&out).unwrap());
        }
        "spec" => {
            // speculative vs plain greedy decoding, one sequence at a time
            let (m, t) = load(&a);
            let draft = Model::load(std::path::Path::new(&a.get("draft", "models/draft.llmw"))).expect("draft model");
            let text = std::fs::read(a.get("text", "")).expect("--text FILE");
            let ids = t.encode(&text);
            let (n, plen, ngen): (usize, usize, usize) = (a.num("requests", 16), a.num("prompt-tokens", 64), a.num("gen-tokens", 128));
            let prompts: Vec<Vec<u32>> = (0..n).map(|i| ids[(i * 1777) % (ids.len() - plen)..][..plen].to_vec()).collect();
            let mut e = Engine::new(
                m.clone(),
                t.clone(),
                EngineConfig {
                    max_seqs: 1,
                    ..engine_cfg(&a, &m)
                },
            );
            let t0 = Instant::now();
            let base: Vec<Vec<u32>> = prompts
                .iter()
                .map(|p| {
                    e.generate(vec![(
                        p.clone(),
                        SamplingParams {
                            max_tokens: ngen,
                            ignore_eos: true,
                            ..Default::default()
                        },
                    )])
                    .remove(0)
                    .0
                })
                .collect();
            let plain = t0.elapsed().as_secs_f64();
            let mut rows = vec![json!({"mode": "plain greedy", "seconds": plain, "tokens_per_s": (n * ngen) as f64 / plain})];
            for k in a.get("k-list", "2,4,6").split(',').map(|x| x.parse::<usize>().unwrap()) {
                let t0 = Instant::now();
                let (mut prop, mut acc, mut tp) = (0, 0, 0);
                for (p, want) in prompts.iter().zip(&base) {
                    let (got, st) = llm_engine::spec::generate(&m, &draft, p, ngen, k);
                    assert_eq!(&got, want, "speculative output differs from greedy");
                    prop += st.proposed;
                    acc += st.accepted;
                    tp += st.target_passes;
                }
                let secs = t0.elapsed().as_secs_f64();
                rows.push(json!({"mode": format!("speculative k={k}"), "seconds": secs, "tokens_per_s": (n * ngen) as f64 / secs,
                                 "speedup": plain / secs, "acceptance": acc as f64 / prop as f64, "target_passes_per_token": tp as f64 / (n * ngen) as f64}));
            }
            println!("{}", serde_json::to_string_pretty(&rows).unwrap());
        }
        _ => {
            eprintln!("usage: llm generate|chat|serve|ppl|bench|prefix-bench|spec|dump|tokenize [--model F] [--tokenizer F] [--q8] ...");
            std::process::exit(2);
        }
    }
}

/// Offline throughput: N requests (prompts cut from --text), greedy, all
/// submitted at once, for several max-seqs settings.
fn bench(a: &Args) {
    let (m, t) = load(a);
    let text = std::fs::read(a.get("text", "")).expect("--text FILE");
    let ids = t.encode(&text);
    let n: usize = a.num("requests", 64);
    let plen: usize = a.num("prompt-tokens", 96);
    let ngen: usize = a.num("gen-tokens", 128);
    let prompts: Vec<Vec<u32>> = (0..n).map(|i| ids[(i * 997) % (ids.len() - plen)..][..plen].to_vec()).collect();
    let mut rows = vec![];
    for seqs in a.get("max-seqs-list", "1,4,16,64").split(',').map(|x| x.parse::<usize>().unwrap()) {
        let cfg = EngineConfig {
            max_seqs: seqs,
            ..engine_cfg(a, &m)
        };
        let mut e = Engine::new(m.clone(), t.clone(), cfg);
        let t0 = Instant::now();
        let out = e.generate(
            prompts
                .iter()
                .map(|p| {
                    (
                        p.clone(),
                        SamplingParams {
                            max_tokens: ngen,
                            ignore_eos: true,
                            ..Default::default()
                        },
                    )
                })
                .collect(),
        );
        let secs = t0.elapsed().as_secs_f64();
        let toks: usize = out.iter().map(|o| o.0.len()).sum();
        // per-sequence decode rate: tokens after the first, over the time after the first
        let dec: f64 = out
            .iter()
            .map(|o| (o.0.len() - 1) as f64 / ((o.2.total_ms - o.2.ttft_ms) / 1e3))
            .sum::<f64>()
            / n as f64;
        let mut ttft: Vec<f64> = out.iter().map(|o| o.2.ttft_ms).collect();
        ttft.sort_by(f64::total_cmp);
        let r = json!({"max_seqs": seqs, "requests": n, "prompt_tokens": plen, "gen_tokens": ngen, "seconds": secs,
                       "output_tokens_per_s": toks as f64 / secs, "total_tokens_per_s": (toks + n * plen) as f64 / secs,
                       "ttft_ms_p50": ttft[n / 2], "ttft_ms_max": ttft[n - 1], "decode_tokens_per_s_per_seq": dec, "steps": e.stats.steps,
                       "mean_batch_seqs": e.stats.decode_tokens as f64 / e.stats.steps as f64, "quantized": a.flag("q8")});
        eprintln!("{r}");
        rows.push(r);
    }
    println!("{}", serde_json::to_string_pretty(&rows).unwrap());
}

/// Logits for every position of each prompt. --input F: a JSON list of
/// token-id lists. Writes little-endian f32 [positions, vocab] for all
/// prompts, concatenated, to --out. The KV blocks are assigned in reverse
/// order, so the paged path is exercised.
fn dump(a: &Args) {
    let (m, _) = load(a);
    let prompts: Vec<Vec<u32>> = serde_json::from_str(&std::fs::read_to_string(a.get("input", "")).expect("--input FILE")).unwrap();
    let mut out = Vec::new();
    let bs = 16;
    for p in &prompts {
        let nb = p.len().div_ceil(bs);
        let mut kv = llm_engine::kv::KvCache::new(m.cfg.layers, m.cfg.kv_heads, m.cfg.head_dim(), nb, bs);
        let blocks: Vec<u32> = (0..nb as u32).rev().collect();
        let toks: Vec<StepToken> = p
            .iter()
            .enumerate()
            .map(|(pos, &token)| StepToken {
                token,
                pos,
                blocks: &blocks,
                want_logits: true,
            })
            .collect();
        for l in m.forward(&toks, &mut kv) {
            for x in l {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
    }
    std::fs::write(a.get("out", "logits.f32"), &out).expect("write --out");
    println!(
        "{}",
        json!({"prompts": prompts.len(), "positions": prompts.iter().map(|p| p.len()).sum::<usize>(), "vocab": m.cfg.vocab})
    );
}

/// Prefix caching: requests that share a long prefix (a "system prompt"),
/// submitted one after another, with and without the prefix cache.
fn prefix_bench(a: &Args) {
    let (m, t) = load(a);
    let text = std::fs::read(a.get("text", "")).expect("--text FILE");
    let ids = t.encode(&text);
    let (n, plen, slen, ngen): (usize, usize, usize, usize) = (
        a.num("requests", 8),
        a.num("prefix-tokens", 512),
        a.num("suffix-tokens", 16),
        a.num("gen-tokens", 16),
    );
    let prefix = &ids[..plen];
    let mut rows = vec![];
    for cache in [false, true] {
        let mut cfg = engine_cfg(a, &m);
        cfg.prefix_caching = cache;
        let mut e = Engine::new(m.clone(), t.clone(), cfg);
        let (mut ttft, mut cached, t0) = (vec![], 0, Instant::now());
        for i in 0..n {
            let mut p = prefix.to_vec();
            let s = plen + i * slen;
            p.extend_from_slice(&ids[s..s + slen]);
            let sp = SamplingParams {
                max_tokens: ngen,
                ignore_eos: true,
                ..Default::default()
            };
            let (_, _, u) = e.generate(vec![(p, sp)]).remove(0);
            ttft.push(u.ttft_ms);
            cached += u.cached_tokens;
        }
        let secs = t0.elapsed().as_secs_f64();
        let r = json!({"prefix_caching": cache, "requests": n, "prefix_tokens": plen, "suffix_tokens": slen, "gen_tokens": ngen,
                       "seconds": secs, "ttft_ms_first": ttft[0], "ttft_ms_mean_rest": ttft[1..].iter().sum::<f64>() / (n - 1) as f64,
                       "cached_prompt_tokens": cached, "prefill_tokens_computed": e.stats.prefill_tokens});
        eprintln!("{r}");
        rows.push(r);
    }
    println!("{}", serde_json::to_string_pretty(&rows).unwrap());
}
