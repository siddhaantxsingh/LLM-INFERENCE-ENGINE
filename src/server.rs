//! OpenAI-style HTTP API over the engine.
//!
//! The engine runs on its own thread: it drains newly submitted requests,
//! runs one `step()` (one batch for every active sequence), and forwards
//! each sequence's events to that request's channel. Handlers either stream
//! the events as server-sent events or collect them into one JSON reply.
//! A client that disconnects drops its channel; the engine notices on the
//! next send and aborts the sequence, freeing its KV blocks.

use crate::engine::{Engine, EngineStats, Event, FinishReason, Usage};
use crate::sampler::SamplingParams;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::json;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::time::Duration;
use tokio::sync::mpsc as tmpsc;

pub enum Msg {
    Add {
        id: u64,
        prompt: Vec<u32>,
        params: SamplingParams,
        tx: tmpsc::UnboundedSender<Event>,
        reply: std::sync::mpsc::Sender<Result<(), String>>,
    },
}

#[derive(Clone)]
pub struct AppState {
    tx: mpsc::Sender<Msg>,
    next_id: Arc<AtomicU64>,
    tok: Arc<crate::tokenizer::Tokenizer>,
    model_name: Arc<String>,
    stats: Arc<Mutex<(EngineStats, usize, usize)>>,
    max_seq: usize,
}

/// Start the engine thread and return the router.
pub fn start(mut engine: Engine, model_name: String) -> Router {
    let (tx, rx) = mpsc::channel::<Msg>();
    let stats = Arc::new(Mutex::new((EngineStats::default(), 0, 0)));
    let st = stats.clone();
    let tok = engine.tok.clone();
    let max_seq = engine.model.cfg.max_seq;
    std::thread::Builder::new()
        .name("engine".into())
        .spawn(move || {
            let mut chans: HashMap<u64, tmpsc::UnboundedSender<Event>> = HashMap::new();
            loop {
                // block when idle, otherwise just drain what has arrived
                let first = if engine.has_work() {
                    rx.try_recv().ok()
                } else {
                    match rx.recv() {
                        Ok(m) => Some(m),
                        Err(_) => return,
                    }
                };
                let mut msgs: Vec<Msg> = first.into_iter().collect();
                msgs.extend(rx.try_iter());
                for Msg::Add {
                    id,
                    prompt,
                    params,
                    tx,
                    reply,
                } in msgs
                {
                    let r = engine.add(id, prompt, params);
                    if r.is_ok() {
                        chans.insert(id, tx);
                    }
                    let _ = reply.send(r);
                }
                for ev in engine.step() {
                    let id = match &ev {
                        Event::Token { id, .. } | Event::Done { id, .. } => *id,
                    };
                    let done = matches!(ev, Event::Done { .. });
                    let alive = chans.get(&id).is_some_and(|c| c.send(ev).is_ok());
                    if done || !alive {
                        chans.remove(&id);
                        if !alive && !done {
                            engine.abort(id);
                        }
                    }
                }
                *st.lock().unwrap() = (engine.stats.clone(), engine.num_running(), engine.num_waiting());
            }
        })
        .expect("spawn engine thread");
    let state = AppState {
        tx,
        next_id: Arc::new(AtomicU64::new(1)),
        tok,
        model_name: Arc::new(model_name),
        stats,
        max_seq,
    };
    Router::new()
        .route("/healthz", get(|| async { "ok" }))
        .route("/v1/models", get(models))
        .route("/v1/completions", post(completions))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/tokenize", post(tokenize))
        .route("/metrics", get(metrics))
        .with_state(state)
}

#[derive(Deserialize)]
pub struct CompletionRequest {
    pub prompt: String,
    #[serde(default)]
    pub stream: bool,
    #[serde(flatten)]
    pub params: SamplingParams,
}

fn err(code: StatusCode, msg: impl Into<String>) -> Response {
    (code, Json(json!({"error": {"message": msg.into()}}))).into_response()
}

async fn models(State(s): State<AppState>) -> Json<serde_json::Value> {
    Json(json!({"object": "list", "data": [{"id": *s.model_name, "object": "model", "max_seq": s.max_seq}]}))
}

async fn tokenize(State(s): State<AppState>, Json(v): Json<serde_json::Value>) -> Response {
    let Some(text) = v["prompt"].as_str() else {
        return err(StatusCode::BAD_REQUEST, "prompt required");
    };
    Json(json!({"tokens": s.tok.encode(text.as_bytes())})).into_response()
}

fn finish_str(r: FinishReason) -> &'static str {
    match r {
        FinishReason::Length => "length",
        FinishReason::Stop => "stop",
    }
}

#[derive(Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Deserialize)]
pub struct ChatRequest {
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: bool,
    #[serde(flatten)]
    pub params: SamplingParams,
}

async fn completions(State(s): State<AppState>, Json(req): Json<CompletionRequest>) -> Response {
    let prompt = s.tok.encode_prompt(&req.prompt);
    run(s, prompt, req.params, req.stream, false).await
}

/// Chat completions: the conversation is rendered with the model's chat
/// template (ChatML only) and generation stops at the end-of-turn token.
async fn chat_completions(State(s): State<AppState>, Json(req): Json<ChatRequest>) -> Response {
    let msgs: Vec<(String, String)> = req.messages.into_iter().map(|m| (m.role, m.content)).collect();
    let text = match s.tok.apply_chat_template(&msgs) {
        Ok(t) => t,
        Err(e) => return err(StatusCode::BAD_REQUEST, e),
    };
    let prompt = s.tok.encode_prompt(&text);
    run(s, prompt, req.params, req.stream, true).await
}

async fn run(s: AppState, prompt: Vec<u32>, params: SamplingParams, stream: bool, chat: bool) -> Response {
    if params.temperature < 0.0 || !(0.0..=1.0).contains(&params.top_p) || params.top_p == 0.0 {
        return err(StatusCode::BAD_REQUEST, "temperature must be >= 0 and top_p in (0, 1]");
    }
    let id = s.next_id.fetch_add(1, Ordering::Relaxed);
    let (tx, mut rx) = tmpsc::unbounded_channel();
    let (rtx, rrx) = std::sync::mpsc::channel();
    if s.tx
        .send(Msg::Add {
            id,
            prompt,
            params,
            tx,
            reply: rtx,
        })
        .is_err()
    {
        return err(StatusCode::SERVICE_UNAVAILABLE, "engine stopped");
    }
    match tokio::task::spawn_blocking(move || rrx.recv()).await {
        Ok(Ok(Ok(()))) => {}
        Ok(Ok(Err(e))) => return err(StatusCode::BAD_REQUEST, e),
        _ => return err(StatusCode::SERVICE_UNAVAILABLE, "engine stopped"),
    }
    let name = s.model_name.clone();
    let cid = format!("cmpl-{id}");
    if stream {
        let stream = async_stream(rx, cid, name, chat);
        return Sse::new(stream)
            .keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
            .into_response();
    }
    let mut text = String::new();
    let mut usage = Usage::default();
    let mut reason = "length";
    while let Some(ev) = rx.recv().await {
        match ev {
            Event::Token { text: t, .. } => text.push_str(&t),
            Event::Done {
                reason: r,
                text: t,
                usage: u,
                ..
            } => {
                text.push_str(&t);
                usage = u;
                reason = finish_str(r);
                break;
            }
        }
    }
    let choice = if chat {
        json!({"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": reason})
    } else {
        json!({"index": 0, "text": text, "finish_reason": reason})
    };
    Json(json!({
        "id": cid, "object": if chat { "chat.completion" } else { "text_completion" }, "model": *name,
        "choices": [choice],
        "usage": {"prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens,
                  "total_tokens": usage.prompt_tokens + usage.completion_tokens, "cached_tokens": usage.cached_tokens,
                  "ttft_ms": usage.ttft_ms, "total_ms": usage.total_ms, "preemptions": usage.preemptions}
    }))
    .into_response()
}

fn async_stream(
    mut rx: tmpsc::UnboundedReceiver<Event>,
    cid: String,
    name: Arc<String>,
    chat: bool,
) -> impl futures::Stream<Item = Result<SseEvent, std::convert::Infallible>> {
    let (otx, orx) = tmpsc::unbounded_channel::<Result<SseEvent, std::convert::Infallible>>();
    tokio::spawn(async move {
        while let Some(ev) = rx.recv().await {
            let (text, reason, usage) = match ev {
                Event::Token { text, .. } => (text, None, None),
                Event::Done { text, reason, usage, .. } => (text, Some(finish_str(reason)), Some(usage)),
            };
            let mut chunk = if chat {
                json!({"id": cid, "object": "chat.completion.chunk", "model": *name,
                       "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": reason}]})
            } else {
                json!({"id": cid, "object": "text_completion", "model": *name,
                       "choices": [{"index": 0, "text": text, "finish_reason": reason}]})
            };
            if let Some(u) = &usage {
                chunk["usage"] = json!({"prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens,
                                        "cached_tokens": u.cached_tokens, "ttft_ms": u.ttft_ms, "total_ms": u.total_ms});
            }
            if otx.send(Ok(SseEvent::default().data(chunk.to_string()))).is_err() {
                return; // client gone: dropping `rx` makes the engine abort the sequence
            }
            if reason.is_some() {
                let _ = otx.send(Ok(SseEvent::default().data("[DONE]")));
                return;
            }
        }
    });
    tokio_stream::wrappers::UnboundedReceiverStream::new(orx)
}

async fn metrics(State(s): State<AppState>) -> String {
    let (st, running, waiting) = s.stats.lock().unwrap().clone();
    let mut o = String::new();
    let mut put = |n: &str, k: &str, v: f64| o.push_str(&format!("# TYPE llm_{n} {k}\nllm_{n} {v}\n"));
    put("steps_total", "counter", st.steps as f64);
    put("prefill_tokens_total", "counter", st.prefill_tokens as f64);
    put("decode_tokens_total", "counter", st.decode_tokens as f64);
    put("preemptions_total", "counter", st.preemptions as f64);
    put("running_sequences", "gauge", running as f64);
    put("waiting_sequences", "gauge", waiting as f64);
    put("kv_blocks_total", "gauge", st.pool.total as f64);
    put("kv_blocks_in_use", "gauge", st.pool.in_use as f64);
    put("kv_blocks_cached_free", "gauge", st.pool.cached_free as f64);
    put("prefix_cache_hits_total", "counter", st.pool.prefix_hits as f64);
    put("prefix_cache_lookups_total", "counter", st.pool.prefix_lookups as f64);
    o
}
