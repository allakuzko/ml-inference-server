# ML Inference Server 🦀

A high-performance sentiment analysis REST API built with Rust + ONNX Runtime.

## Stack
- **Rust** + **axum** — async web server
- **ort** — ONNX Runtime bindings
- **tokenizers** — HuggingFace tokenizer
- **anyhow** — error handling
- **Model** — distilbert-base-uncased-finetuned-sst-2-english

## Features
- ✅ Single text inference — `POST /predict`
- ✅ Batch inference — `POST /predict/batch`
- ✅ Health check — `GET /health`
- ✅ Inference time logging
- ✅ Proper error handling
- ✅ Rate limiting (10 req/sec per IP)

## Setup

### 1. Download ONNX Runtime
```bash
wget https://github.com/microsoft/onnxruntime/releases/download/v1.24.2/onnxruntime-linux-x64-1.24.2.tgz
tar -xzf onnxruntime-linux-x64-1.24.2.tgz
sudo cp onnxruntime-linux-x64-1.24.2/lib/* /usr/local/lib/
sudo ldconfig
```

### 2. Download model
```bash
pip install huggingface_hub
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('distilbert-base-uncased-finetuned-sst-2-english', local_dir='model')
"
```

### 3. Run
```bash
export ORT_DYLIB_PATH=/usr/local/lib/libonnxruntime.so.1.24.2
cargo run --release
```

## API

### `POST /predict`
Single text sentiment analysis.

**Request:**
```json
{"text": "I love this movie!"}
```

**Response:**
```json
{
  "text": "I love this movie!",
  "label": "POSITIVE",
  "score": 0.9998,
  "inference_ms": 12
}
```

### `POST /predict/batch`
Batch sentiment analysis (max 32 texts).

**Request:**
```json
{
  "texts": [
    "I love this!",
    "This is terrible.",
    "Pretty good actually."
  ]
}
```

**Response:**
```json
{
  "results": [
    {"text": "I love this!", "label": "POSITIVE", "score": 0.9998, "inference_ms": 12},
    {"text": "This is terrible.", "label": "NEGATIVE", "score": 0.9995, "inference_ms": 10},
    {"text": "Pretty good actually.", "label": "POSITIVE", "score": 0.9987, "inference_ms": 11}
  ],
  "total_ms": 33
}
```

### `GET /health`
Health check.

**Response:**
```json
{"status": "ok", "version": "0.1.0"}
```

## Rate Limiting
- Max **10 requests per second** per IP
- Exceeding the limit returns `429 Too Many Requests`

## Error Handling
All errors return a JSON response:
```json
{"error": "error description", "code": 500}
```
