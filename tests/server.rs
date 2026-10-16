//! HTTP server end to end: real sockets, concurrent clients, streaming.

use futures::StreamExt;
use llm_engine::engine::{Engine, EngineConfig};
use llm_engine::model::Model;
use llm_engine::sampler::SamplingParams;
use llm_engine::tokenizer::Tokenizer;
use serde_json::{Value, json};
use std::sync::Arc;

fn root() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn parts() -> (Arc<Model>, Arc<Tokenizer>) {
    let m = Arc::new(Model::load(&root().join("models/small.llmw")).unwrap());
    let t = Arc::new(Tokenizer::from_json(&std::fs::read_to_string(root().join("models/tokenizer.json")).unwrap()).unwrap());
    (m, t)
}

async fn serve(cfg: EngineConfig) -> String {
    let (m, t) = parts();
    let app = llm_engine::server::start(Engine::new(m, t, cfg), "small".into());
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", l.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(l, app).await.unwrap() });
    url
}

async fn metric(c: &reqwest::Client, url: &str, name: &str) -> f64 {
    let m = c.get(format!("{url}/metrics")).send().await.unwrap().text().await.unwrap();
    m.lines()
        .find_map(|l| l.strip_prefix(&format!("{name} ")))
        .unwrap()
        .parse()
        .unwrap()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn completions_streaming_and_concurrency() {
    let url = serve(EngineConfig::default()).await;
    let c = reqwest::Client::new();
    let prompts = [
        "def fibonacci(n):\n    ",
        "import os\n",
        "class Stack:\n    def push(self, item):\n        ",
        "    return ",
    ];

    // what the engine produces offline for the same requests
    let (m, t) = parts();
    let mut offline = Engine::new(m, t.clone(), EngineConfig::default());
    let want: Vec<String> = offline
        .generate(
            prompts
                .iter()
                .map(|p| {
                    (
                        t.encode(p.as_bytes()),
                        SamplingParams {
                            max_tokens: 24,
                            ..Default::default()
                        },
                    )
                })
                .collect(),
        )
        .into_iter()
        .map(|o| o.1)
        .collect();

    // 4 concurrent non-streaming requests, batched together by the engine
    let futs = prompts.iter().map(|p| {
        let c = c.clone();
        let url = url.clone();
        let p = p.to_string();
        async move {
            let r = c
                .post(format!("{url}/v1/completions"))
                .json(&json!({"prompt": p, "max_tokens": 24}))
                .send()
                .await
                .unwrap();
            assert_eq!(r.status(), 200);
            r.json::<Value>().await.unwrap()
        }
    });
    let got: Vec<Value> = futures::future::join_all(futs).await;
    for (g, w) in got.iter().zip(&want) {
        assert_eq!(g["choices"][0]["text"], *w);
        assert_eq!(g["usage"]["completion_tokens"], 24);
        assert_eq!(g["choices"][0]["finish_reason"], "length");
    }

    // streaming: chunks concatenate to the same text, then [DONE]
    let r = c
        .post(format!("{url}/v1/completions"))
        .json(&json!({"prompt": prompts[0], "max_tokens": 24, "stream": true}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    assert!(r.headers()["content-type"].to_str().unwrap().starts_with("text/event-stream"));
    let body = String::from_utf8(r.bytes().await.unwrap().to_vec()).unwrap();
    let mut text = String::new();
    let mut chunks = 0;
    let mut done = false;
    for line in body.lines().filter_map(|l| l.strip_prefix("data: ")) {
        if line == "[DONE]" {
            done = true;
            continue;
        }
        let v: Value = serde_json::from_str(line).unwrap();
        text.push_str(v["choices"][0]["text"].as_str().unwrap());
        chunks += 1;
    }
    assert!(done && chunks >= 2, "{chunks} chunks");
    assert_eq!(text, want[0]);

    // stop strings, validation
    let r: Value = c
        .post(format!("{url}/v1/completions"))
        .json(&json!({"prompt": prompts[0], "max_tokens": 24, "stop": [want[0].chars().nth(3).unwrap().to_string()]}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(r["choices"][0]["finish_reason"], "stop");
    assert!(want[0].starts_with(r["choices"][0]["text"].as_str().unwrap()));
    for bad in [
        json!({"prompt": "", "max_tokens": 4}),
        json!({"prompt": "x", "temperature": -1.0}),
        json!({"prompt": "x", "top_p": 0.0}),
    ] {
        let r = c.post(format!("{url}/v1/completions")).json(&bad).send().await.unwrap();
        assert_eq!(r.status(), 400, "{bad}");
    }
    let m: Value = c.get(format!("{url}/v1/models")).send().await.unwrap().json().await.unwrap();
    assert_eq!(m["data"][0]["id"], "small");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn disconnected_stream_is_aborted() {
    let url = serve(EngineConfig::default()).await;
    let c = reqwest::Client::new();
    let r = c
        .post(format!("{url}/v1/completions"))
        .json(&json!({"prompt": "def f(x):\n", "max_tokens": 400, "stream": true}))
        .send()
        .await
        .unwrap();
    let mut s = r.bytes_stream();
    let _ = s.next().await; // first chunk arrived: the sequence is running
    drop(s);
    let t0 = std::time::Instant::now();
    loop {
        if metric(&c, &url, "llm_running_sequences").await == 0.0 && metric(&c, &url, "llm_kv_blocks_in_use").await == 0.0 {
            break;
        }
        assert!(t0.elapsed().as_secs() < 10, "sequence not aborted after disconnect");
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    assert!(
        metric(&c, &url, "llm_decode_tokens_total").await < 390.0,
        "stopped early, not run to completion"
    );
}
