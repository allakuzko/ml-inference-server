use axum::{routing::{post, get}, Json, Router};
use ort::{
    inputs,
    session::{builder::GraphOptimizationLevel, Session},
    value::Tensor,
};
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tracing::info;

#[derive(Deserialize)]
struct InferenceRequest {
    text: String,
}

#[derive(Deserialize)]
struct BatchInferenceRequest {
    texts: Vec<String>,
}

#[derive(Serialize)]
struct InferenceResponse {
    text: String,
    label: String,
    score: f32,
    inference_ms: u128,
}

#[derive(Serialize)]
struct BatchInferenceResponse {
    results: Vec<InferenceResponse>,
    total_ms: u128,
}

#[derive(Serialize)]
struct HealthResponse {
    status: String,
    version: String,
}

struct AppState {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let session = Session::builder()
        .unwrap()
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .unwrap()
        .with_intra_threads(4)
        .unwrap()
        .commit_from_file("model/onnx/model.onnx")
        .unwrap();

    let tokenizer = Tokenizer::from_file("model/onnx/tokenizer.json").unwrap();

    let state = Arc::new(AppState {
        session: Mutex::new(session),
        tokenizer,
    });

    let app = Router::new()
        .route("/predict", post(predict))
        .route("/predict/batch", post(predict_batch))
        .route("/health", get(health))
        .with_state(state);

    let addr = "0.0.0.0:3000";
    info!("Server running on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

fn run_inference(
    session: &mut Session,
    tokenizer: &Tokenizer,
    text: &str,
) -> InferenceResponse {
    let encoding = tokenizer.encode(text, true).unwrap();
    let len = encoding.get_ids().len();

    let ids: Vec<i64> = encoding.get_ids().iter().map(|x| *x as i64).collect();
    let mask: Vec<i64> = encoding.get_attention_mask().iter().map(|x| *x as i64).collect();

    let input_ids = Tensor::<i64>::from_array(([1, len], ids)).unwrap();
    let attention_mask = Tensor::<i64>::from_array(([1, len], mask)).unwrap();

    let start = Instant::now();

    let outputs = session
        .run(inputs![
            "input_ids" => input_ids,
            "attention_mask" => attention_mask
        ])
        .unwrap();

    let inference_ms = start.elapsed().as_millis();

    let (_, logits_data) = outputs["logits"]
        .try_extract_tensor::<f32>()
        .unwrap();

    let neg = logits_data[0];
    let pos = logits_data[1];

    let label = if pos > neg { "POSITIVE" } else { "NEGATIVE" };
    let score = if pos > neg { pos } else { neg };

    InferenceResponse {
        text: text.to_string(),
        label: label.to_string(),
        score,
        inference_ms,
    }
}

async fn predict(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
    Json(payload): Json<InferenceRequest>,
) -> Json<InferenceResponse> {
    let mut session = state.session.lock().unwrap();
    let result = run_inference(&mut session, &state.tokenizer, &payload.text);

    info!(
        text = %result.text,
        label = %result.label,
        score = %result.score,
        inference_ms = %result.inference_ms,
        "predict"
    );

    Json(result)
}

async fn predict_batch(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
    Json(payload): Json<BatchInferenceRequest>,
) -> Json<BatchInferenceResponse> {
    let total_start = Instant::now();
    let mut session = state.session.lock().unwrap();

    let results: Vec<InferenceResponse> = payload.texts
        .iter()
        .map(|text| run_inference(&mut session, &state.tokenizer, text))
        .collect();

    let total_ms = total_start.elapsed().as_millis();

    info!(
        count = %results.len(),
        total_ms = %total_ms,
        "predict_batch"
    );

    Json(BatchInferenceResponse { results, total_ms })
}