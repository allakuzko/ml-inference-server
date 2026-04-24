# ML Inference Server 🦀

A high-performance sentiment analysis REST API built with Rust + ONNX Runtime.

## Stack
- **Rust** + **axum** — async web server
- **ort** — ONNX Runtime bindings
- **tokenizers** — HuggingFace tokenizer
- **Model** — distilbert-base-uncased-finetuned-sst-2-english

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

## Usage
```bash
curl -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this!"}'
```

## Response
```json
{"text":"I love this!","label":"POSITIVE","score":0.99}
```
