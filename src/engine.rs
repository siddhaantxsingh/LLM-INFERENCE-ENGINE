//! The engine: continuous batching over a paged KV cache.
//!
//! Every `step()` builds one batch from all running sequences:
//! 1. sequences that are decoding contribute their one pending token;
//! 2. sequences still prefilling contribute a chunk of their prompt;
//! 3. waiting requests are admitted while there is token budget, sequence
//!    room and KV space. On admission the longest cached prefix of the
//!    prompt (whole blocks) is reused instead of recomputed.
//!
//! When KV blocks run out, the most recently admitted sequence is preempted:
//! its blocks are released and it goes back to the front of the queue with
//! everything it has generated so far, to be recomputed later. Because the
//! kernels are batch-invariant (see `ops`), recomputation reproduces the
//! same logits bit for bit, so preemption never changes the output.

use crate::kv::{BlockPool, KvCache, PoolStats, block_hash};
use crate::model::{Model, StepToken};
use crate::sampler::{Rng, SamplingParams, sample};
use crate::tokenizer::Tokenizer;
use serde::Serialize;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Instant;

#[derive(Clone, Debug, Serialize)]
pub struct EngineConfig {
    pub block_size: usize,
    pub num_blocks: usize,
    /// tokens (prefill + decode) per step
    pub max_batch_tokens: usize,
    pub max_seqs: usize,
    pub prefix_caching: bool,
    /// overwrite freed KV blocks with NaN (testing aid)
    pub poison_freed: bool,
}

impl Default for EngineConfig {
    fn default() -> Self {
        EngineConfig {
            block_size: 16,
            num_blocks: 2048,
            max_batch_tokens: 256,
            max_seqs: 64,
            prefix_caching: true,
            poison_freed: false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Length,
    Stop,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    /// prompt tokens served from the prefix cache
    pub cached_tokens: usize,
    pub preemptions: u32,
    pub ttft_ms: f64,
    pub total_ms: f64,
}

#[derive(Clone, Debug)]
pub enum Event {
    Token {
        id: u64,
        token: u32,
        text: String,
    },
    Done {
        id: u64,
        reason: FinishReason,
        text: String,
        usage: Usage,
    },
}

#[derive(Default, Debug, Clone, Serialize)]
pub struct EngineStats {
    pub steps: u64,
    pub prefill_tokens: u64,
    pub decode_tokens: u64,
    pub preemptions: u64,
    pub max_batch_seqs: usize,
    pub pool: PoolStats,
}

struct Seq {
    id: u64,
    tokens: Vec<u32>,
    prompt_len: usize,
    computed: usize,
    blocks: Vec<u32>,
    /// full blocks registered in (or taken from) the prefix cache
    hashed: usize,
    last_hash: u64,
    params: SamplingParams,
    rng: Rng,
    max_tokens: usize,
    emitted: usize,
    cached_tokens: usize,
    preemptions: u32,
    arrival: Instant,
    first_token: Option<Instant>,
}

pub struct Engine {
    pub model: Arc<Model>,
    pub tok: Arc<Tokenizer>,
    pub cfg: EngineConfig,
    kv: KvCache,
    pool: BlockPool,
    waiting: VecDeque<Seq>,
    running: Vec<Seq>,
    pub stats: EngineStats,
}

impl Engine {
    pub fn new(model: Arc<Model>, tok: Arc<Tokenizer>, cfg: EngineConfig) -> Engine {
        let c = &model.cfg;
        let mut kv = KvCache::new(c.layers, c.kv_heads, c.head_dim(), cfg.num_blocks, cfg.block_size);
        if cfg.poison_freed {
            for b in 0..cfg.num_blocks {
                kv.poison(b);
            }
        }
        Engine {
            pool: BlockPool::new(cfg.num_blocks),
            kv,
            model,
            tok,
            cfg,
            waiting: VecDeque::new(),
            running: vec![],
            stats: Default::default(),
        }
    }

    pub fn kv_bytes(&self) -> usize {
        self.kv.bytes()
    }

    pub fn has_work(&self) -> bool {
        !self.waiting.is_empty() || !self.running.is_empty()
    }

    pub fn num_running(&self) -> usize {
        self.running.len()
    }
    pub fn num_waiting(&self) -> usize {
        self.waiting.len()
    }

    /// Queue a request. Fails if the prompt is empty or cannot fit.
    pub fn add(&mut self, id: u64, prompt: Vec<u32>, params: SamplingParams) -> Result<(), String> {
        let max_seq = self.model.cfg.max_seq;
        if prompt.is_empty() {
            return Err("empty prompt".into());
        }
        if prompt.len() >= max_seq {
            return Err(format!("prompt has {} tokens; the model's context is {max_seq}", prompt.len()));
        }
        if let Some(&t) = prompt.iter().find(|&&t| t as usize >= self.model.cfg.vocab) {
            return Err(format!("token {t} out of vocabulary"));
        }
        let max_tokens = params.max_tokens.min(max_seq - prompt.len());
        let need = (prompt.len() + max_tokens).div_ceil(self.cfg.block_size);
        if need > self.cfg.num_blocks {
            return Err(format!("request needs {need} KV blocks; the pool has {}", self.cfg.num_blocks));
        }
        self.waiting.push_back(Seq {
            id,
            prompt_len: prompt.len(),
            tokens: prompt,
            computed: 0,
            blocks: vec![],
            hashed: 0,
            last_hash: 0,
            rng: Rng::new(params.seed),
            params,
            max_tokens,
            emitted: 0,
            cached_tokens: 0,
            preemptions: 0,
            arrival: Instant::now(),
            first_token: None,
        });
        Ok(())
    }

    /// Cancel a request (e.g. the client went away).
    pub fn abort(&mut self, id: u64) {
        if let Some(i) = self.running.iter().position(|s| s.id == id) {
            let s = self.running.remove(i);
            self.free_seq(s);
        }
        self.waiting.retain(|s| s.id != id);
        self.stats.pool = self.pool.snapshot();
    }

    fn release_blocks(&mut self, blocks: &[u32]) {
        for &b in blocks {
            if self.pool.release(b) && self.cfg.poison_freed {
                self.kv.poison(b as usize);
            }
        }
    }

    fn free_seq(&mut self, s: Seq) {
        self.release_blocks(&s.blocks);
    }

    fn preempt(&mut self, idx: usize) {
        let mut s = self.running.remove(idx);
        let blocks = std::mem::take(&mut s.blocks);
        self.release_blocks(&blocks);
        s.computed = 0;
        s.hashed = 0;
        s.last_hash = 0;
        s.preemptions += 1;
        self.stats.preemptions += 1;
        self.waiting.push_front(s);
    }

    /// Ensure `s` has blocks for positions < upto. False if the pool is empty.
    fn grow(pool: &mut BlockPool, s: &mut Seq, upto: usize, bs: usize) -> bool {
        while s.blocks.len() * bs < upto {
            match pool.alloc() {
                Some(b) => s.blocks.push(b),
                None => return false,
            }
        }
        true
    }

    /// Reuse cached blocks for the prompt's leading full blocks. The block
    /// holding the last token is always computed, so a fully cached prompt
    /// still produces logits without writing into a shared block.
    fn match_prefix(&mut self, s: &mut Seq) {
        if !self.cfg.prefix_caching || s.computed != 0 {
            return;
        }
        let bs = self.cfg.block_size;
        let full = (s.tokens.len() - 1) / bs;
        let mut h = 0;
        for i in 0..full {
            h = block_hash(h, &s.tokens[i * bs..(i + 1) * bs]);
            match self.pool.lookup(h) {
                Some(b) => {
                    s.blocks.push(b);
                    s.hashed = i + 1;
                    s.last_hash = h;
                }
                None => break,
            }
        }
        s.computed = s.hashed * bs;
        s.cached_tokens = s.computed.min(s.prompt_len);
    }

    pub fn step(&mut self) -> Vec<Event> {
        let bs = self.cfg.block_size;
        let mut budget = self.cfg.max_batch_tokens;
        // (index into running, start, len)
        let mut plan: Vec<(usize, usize, usize)> = vec![];

        // 1+2: running sequences, oldest first; decoding ones have 1 token left
        let mut order: Vec<usize> = (0..self.running.len()).collect();
        order.sort_by_key(|&i| (self.running[i].tokens.len() - self.running[i].computed > 1, self.running[i].arrival));
        let mut i = 0;
        while i < order.len() && budget > 0 {
            let idx = order[i];
            let (start, want) = {
                let s = &self.running[idx];
                (s.computed, (s.tokens.len() - s.computed).min(budget))
            };
            let ok = Self::grow(&mut self.pool, &mut self.running[idx], start + want, bs);
            if !ok {
                // preempt the newest sequence that is not already planned
                let planned: Vec<usize> = plan.iter().map(|p| p.0).collect();
                let victim = (0..self.running.len())
                    .filter(|j| !planned.contains(j))
                    .max_by_key(|&j| self.running[j].arrival);
                match victim {
                    Some(v) => {
                        self.preempt(v);
                        // indices shift: rebuild order and plan positions
                        let fix = |x: usize| if x > v { x - 1 } else { x };
                        for p in plan.iter_mut() {
                            p.0 = fix(p.0);
                        }
                        order = order.into_iter().filter(|&x| x != v).map(fix).collect();
                        if v == idx {
                            continue; // it was this one; `order[i]` is now the next
                        }
                        continue; // retry this sequence
                    }
                    None => break, // everything left is planned; try again next step
                }
            }
            plan.push((idx, start, want));
            budget -= want;
            i += 1;
        }

        // 3: admit waiting requests
        while budget > 0 && self.running.len() < self.cfg.max_seqs {
            let Some(mut s) = self.waiting.pop_front() else { break };
            self.match_prefix(&mut s);
            let want = (s.tokens.len() - s.computed).min(budget);
            let upto = s.computed + want;
            if !Self::grow(&mut self.pool, &mut s, upto, bs) {
                // not enough KV space now: undo and wait for running work to finish
                self.free_seq_keep(&mut s);
                self.waiting.push_front(s);
                break;
            }
            self.running.push(s);
            plan.push((self.running.len() - 1, self.running.last().unwrap().computed, want));
            budget -= want;
        }
        if plan.is_empty() {
            return vec![];
        }

        // build and run the batch
        let mut toks: Vec<StepToken> = vec![];
        for &(idx, start, len) in &plan {
            let s = &self.running[idx];
            for p in start..start + len {
                toks.push(StepToken {
                    token: s.tokens[p],
                    pos: p,
                    blocks: &s.blocks,
                    want_logits: p + 1 == s.tokens.len(),
                });
            }
        }
        let decode = plan
            .iter()
            .filter(|&&(idx, start, len)| {
                let s = &self.running[idx];
                len == 1 && start + 1 == s.tokens.len() && s.tokens.len() > s.prompt_len
            })
            .count();
        self.stats.decode_tokens += decode as u64;
        self.stats.prefill_tokens += (toks.len() - decode) as u64;
        self.stats.steps += 1;
        self.stats.max_batch_seqs = self.stats.max_batch_seqs.max(plan.len());
        let logits = self.model.forward(&toks, &mut self.kv);
        drop(toks);

        // advance, register full blocks, sample
        let mut events = vec![];
        let mut li = 0;
        let mut finished = vec![];
        for &(idx, start, len) in &plan {
            let s = &mut self.running[idx];
            s.computed = start + len;
            if self.cfg.prefix_caching {
                while s.hashed < s.computed / bs {
                    let i = s.hashed;
                    s.last_hash = block_hash(s.last_hash, &s.tokens[i * bs..(i + 1) * bs]);
                    self.pool.register(s.blocks[i], s.last_hash);
                    s.hashed += 1;
                }
            }
            if s.computed < s.tokens.len() {
                continue; // prefill continues next step
            }
            let next = sample(&logits[li], &s.params, &mut s.rng);
            li += 1;
            s.tokens.push(next);
            s.first_token.get_or_insert_with(Instant::now);
            let generated = s.tokens.len() - s.prompt_len;
            // the model's end-of-sequence token ends the request; it adds no text
            let eos = !s.params.ignore_eos && self.tok.stop_ids.contains(&next);
            let text = self.tok.decode(&s.tokens[s.prompt_len..s.tokens.len() - eos as usize]);
            let mut reason = eos.then_some(FinishReason::Stop);
            let mut end = text.len();
            for stop in &s.params.stop {
                if !stop.is_empty()
                    && let Some(pos) = find(&text, stop.as_bytes())
                {
                    end = end.min(pos);
                    reason = Some(FinishReason::Stop);
                }
            }
            if reason.is_none() && generated >= s.max_tokens {
                reason = Some(FinishReason::Length);
            }
            // hold back bytes that may be the start of a stop string or of a UTF-8 character
            let hold = if reason.is_some() {
                0
            } else {
                s.params.stop.iter().map(|x| x.len().saturating_sub(1)).max().unwrap_or(0)
            };
            let upto = valid_utf8_prefix(&text[..end.saturating_sub(hold).max(s.emitted)]).max(s.emitted);
            let delta = String::from_utf8_lossy(&text[s.emitted..upto]).into_owned();
            s.emitted = upto;
            match reason {
                None => events.push(Event::Token {
                    id: s.id,
                    token: next,
                    text: delta,
                }),
                Some(r) => {
                    let rest = String::from_utf8_lossy(&text[s.emitted.min(end)..end]).into_owned();
                    let now = Instant::now();
                    let usage = Usage {
                        prompt_tokens: s.prompt_len,
                        completion_tokens: generated,
                        cached_tokens: s.cached_tokens,
                        preemptions: s.preemptions,
                        ttft_ms: (s.first_token.unwrap() - s.arrival).as_secs_f64() * 1e3,
                        total_ms: (now - s.arrival).as_secs_f64() * 1e3,
                    };
                    events.push(Event::Token {
                        id: s.id,
                        token: next,
                        text: delta,
                    });
                    events.push(Event::Done {
                        id: s.id,
                        reason: r,
                        text: rest,
                        usage,
                    });
                    finished.push(idx);
                }
            }
        }
        finished.sort_unstable();
        for idx in finished.into_iter().rev() {
            let s = self.running.remove(idx);
            self.free_seq(s);
        }
        self.stats.pool = self.pool.snapshot(); // after the releases above
        events
    }

    fn free_seq_keep(&mut self, s: &mut Seq) {
        let blocks = std::mem::take(&mut s.blocks);
        self.release_blocks(&blocks);
        s.computed = 0;
        s.hashed = 0;
        s.last_hash = 0;
        s.cached_tokens = 0;
    }

    pub fn pool_check(&self) -> Result<(), String> {
        self.pool.check()?;
        let held: usize = self.running.iter().map(|s| s.blocks.len()).sum();
        let in_use = self.pool.snapshot().in_use;
        if held < in_use {
            return Err(format!("{in_use} blocks referenced but sequences hold {held}"));
        }
        Ok(())
    }

    /// Run requests to completion (offline use and tests). Returns, per
    /// request, the generated tokens, text and usage.
    pub fn generate(&mut self, reqs: Vec<(Vec<u32>, SamplingParams)>) -> Vec<(Vec<u32>, String, Usage)> {
        let base = 1_000_000_000 + self.stats.steps * 1000;
        let n = reqs.len();
        let mut out: Vec<(Vec<u32>, String, Option<Usage>)> = vec![(vec![], String::new(), None); n];
        for (i, (p, sp)) in reqs.into_iter().enumerate() {
            self.add(base + i as u64, p, sp).expect("request accepted");
        }
        while out.iter().any(|o| o.2.is_none()) {
            for e in self.step() {
                match e {
                    Event::Token { id, token, text } => {
                        let o = &mut out[(id - base) as usize];
                        o.0.push(token);
                        o.1.push_str(&text);
                    }
                    Event::Done { id, text, usage, .. } => {
                        let o = &mut out[(id - base) as usize];
                        o.1.push_str(&text);
                        o.2 = Some(usage);
                    }
                }
            }
        }
        out.into_iter().map(|(t, s, u)| (t, s, u.unwrap())).collect()
    }
}

fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

fn valid_utf8_prefix(b: &[u8]) -> usize {
    match std::str::from_utf8(b) {
        Ok(_) => b.len(),
        Err(e) => e.valid_up_to(),
    }
}
