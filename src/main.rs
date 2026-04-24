use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use ort::{
    inputs,
    session::{builder::GraphOptimizationLevel, Session},
    value::Tensor,
};
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tracing::{error, info};

// --- Типи помилок ---

struct AppError(anyhow::Error);

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        error!("Request failed: {}", self.0);
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": self.0.to_string(),
                "code": 500
            })),
        )
            .into_response()
    }
}

impl<E: Into<anyhow::Error>> From<E> for AppError {
    fn from(e: E) -> Self {
        AppError(e.into())
    }
}

type AppResult<T> = Result<Json<T>, AppError>;

// --- Структури запитів/відповідей ---

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

// --- Стан програми ---

struct AppState {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
}

// --- Main ---

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let session = Session::builder()
        .expect("Failed to create session builder")
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .expect("Failed to set optimization level")
        .with_intra_threads(4)
        .expect("Failed to set intra threads")
        .commit_from_file("model/onnx/model.onnx")
        .expect("Failed to load model — check that model/onnx/model.onnx exists");

    let tokenizer = Tokenizer::from_file("model/onnx/tokenizer.json")
        .expect("Failed to load tokenizer — check that model/onnx/tokenizer.json exists");

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

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("Failed to bind to address");

    axum::serve(listener, app)
        .await
        .expect("Server error");
}

// --- Handlers ---

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
) -> anyhow::Result<InferenceResponse> {
    let encoding = tokenizer
        .encode(text, true)
        .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

    let len = encoding.get_ids().len();

    let ids: Vec<i64> = encoding.get_ids().iter().map(|x| *x as i64).collect();
    let mask: Vec<i64> = encoding.get_attention_mask().iter().map(|x| *x as i64).collect();

    let input_ids = Tensor::<i64>::from_array(([1, len], ids))
        .map_err(|e| anyhow::anyhow!("Failed to create input_ids tensor: {}", e))?;

    let attention_mask = Tensor::<i64>::from_array(([1, len], mask))
        .map_err(|e| anyhow::anyhow!("Failed to create attention_mask tensor: {}", e))?;

    let start = Instant::now();

    let outputs = session
        .run(inputs![
            "input_ids" => input_ids,
            "attention_mask" => attention_mask
        ])
        .map_err(|e| anyhow::anyhow!("Inference failed: {}", e))?;

    let inference_ms = start.elapsed().as_millis();

    let (_, logits_data) = outputs["logits"]
        .try_extract_tensor::<f32>()
        .map_err(|e| anyhow::anyhow!("Failed to extract logits: {}", e))?;

    let neg = logits_data[0];
    let pos = logits_data[1];

    let label = if pos > neg { "POSITIVE" } else { "NEGATIVE" };
    let score = if pos > neg { pos } else { neg };

    Ok(InferenceResponse {
        text: text.to_string(),
        label: label.to_string(),
        score,
        inference_ms,
    })
}

async fn predict(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
    Json(payload): Json<InferenceRequest>,
) -> AppResult<InferenceResponse> {
    let mut session = state
        .session
        .lock()
        .map_err(|e| anyhow::anyhow!("Failed to lock session: {}", e))?;

    let result = run_inference(&mut session, &state.tokenizer, &payload.text)?;

    info!(
        text = %result.text,
        label = %result.label,
        score = %result.score,
        inference_ms = %result.inference_ms,
        "predict"
    );

    Ok(Json(result))
}

async fn predict_batch(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
    Json(payload): Json<BatchInferenceRequest>,
) -> AppResult<BatchInferenceResponse> {
    if payload.texts.is_empty() {
        return Err(AppError(anyhow::anyhow!("texts array cannot be empty")));
    }

    if payload.texts.len() > 32 {
        return Err(AppError(anyhow::anyhow!("Maximum batch size is 32")));
    }

    let total_start = Instant::now();

    let mut session = state
        .session
        .lock()
        .map_err(|e| anyhow::anyhow!("Failed to lock session: {}", e))?;

    let mut results = Vec::new();
    for text in &payload.texts {
        let result = run_inference(&mut session, &state.tokenizer, text)?;
        results.push(result);
    }

    let total_ms = total_start.elapsed().as_millis();

    info!(
        count = %results.len(),
        total_ms = %total_ms,
        "predict_batch"
    );

    Ok(Json(BatchInferenceResponse { results, total_ms }))
}