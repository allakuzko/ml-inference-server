use axum::{routing::{post, get}, Json, Router};
use ort::{
    inputs,
    session::{builder::GraphOptimizationLevel, Session},
    value::Tensor,
};
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;
use std::sync::{Arc, Mutex};
use tracing::info;

#[derive(Deserialize)]
struct InferenceRequest {
    text: String,
}

#[derive(Serialize)]
struct InferenceResponse {
    text: String,
    label: String,
    score: f32,
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

async fn predict(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
    Json(payload): Json<InferenceRequest>,
) -> Json<InferenceResponse> {
    let encoding = state
        .tokenizer
        .encode(payload.text.clone(), true)
        .unwrap();

    let len = encoding.get_ids().len();

    let ids: Vec<i64> = encoding.get_ids().iter().map(|x| *x as i64).collect();
    let mask: Vec<i64> = encoding.get_attention_mask().iter().map(|x| *x as i64).collect();

    let input_ids = Tensor::<i64>::from_array(([1, len], ids)).unwrap();
    let attention_mask = Tensor::<i64>::from_array(([1, len], mask)).unwrap();

    let mut session = state.session.lock().unwrap();

    let outputs = session
        .run(inputs![
            "input_ids" => input_ids,
            "attention_mask" => attention_mask
        ])
        .unwrap();

    let (_, logits_data) = outputs["logits"]
        .try_extract_tensor::<f32>()
        .unwrap();

    let neg = logits_data[0];
    let pos = logits_data[1];

    let label = if pos > neg { "POSITIVE" } else { "NEGATIVE" };
    let score = if pos > neg { pos } else { neg };

    Json(InferenceResponse {
        text: payload.text,
        label: label.to_string(),
        score,
    })
}