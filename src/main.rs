use axum::{
    http::{StatusCode, Request},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
    middleware::{self, Next},
};
use ort::{
    inputs,
    session::{builder::GraphOptimizationLevel, Session},
    value::Tensor,
};
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Instant, Duration};
use tracing::{error, info, warn};

// --- Rate Limiter ---

struct RateLimiter {
    requests: HashMap<String, Vec<Instant>>,
    max_requests: usize,
    window: Duration,
}

impl RateLimiter {
    fn new(max_requests: usize, window: Duration) -> Self {
        Self {
            requests: HashMap::new(),
            max_requests,
            window,
        }
    }

    fn is_allowed(&mut self, ip: &str) -> bool {
        let now = Instant::now();
        let window = self.window;

        let timestamps = self.requests.entry(ip.to_string()).or_default();

        // Видаляємо старі запити поза вікном
        timestamps.retain(|t| now.duration_since(*t) < window);

        if timestamps.len() < self.max_requests {
            timestamps.push(now);
            true
        } else {
            false
        }
    }
}

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
    rate_limiter: Mutex<RateLimiter>,
}

// --- Middleware ---

async fn rate_limit_middleware(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
    req: Request<axum::body::Body>,
    next: Next,
) -> Response {
    let ip = req
        .headers()
        .get("x-forwarded-for")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("unknown")
        .to_string();

    let allowed = state
        .rate_limiter
        .lock()
        .unwrap()
        .is_allowed(&ip);

    if !allowed {
        warn!(ip = %ip, "Rate limit exceeded");
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(serde_json::json!({
                "error": "Too many requests. Max 10 requests per second.",
                "code": 429
            })),
        )
            .into_response();
    }

    next.run(req).await
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
        .expect("Failed to load model");

    let tokenizer = Tokenizer::from_file("model/onnx/tokenizer.json")
        .expect("Failed to load tokenizer");

    let state = Arc::new(AppState {
        session: Mutex::new(session),
        tokenizer,
        rate_limiter: Mutex::new(RateLimiter::new(10, Duration::from_secs(1))),
    });

    let app = Router::new()
        .route("/predict", post(predict))
        .route("/predict/batch", post(predict_batch))
        .route("/health", get(health))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            rate_limit_middleware,
        ))
        .with_state(state);

    let addr = "0.0.0.0:3000";
    info!("Server running on http://{}", addr);
    info!("Rate limit: 10 requests/second per IP");

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
        .unwrap();

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