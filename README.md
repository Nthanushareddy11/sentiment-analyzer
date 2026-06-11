# 🧠 Sentiment + Intent Analyzer API

Real-time NLP pipeline for **sentiment analysis**, **intent classification**, and **NLG response generation** using HuggingFace Transformers.

---

## 🎯 What it does

| Task | Type | Model |
|------|------|-------|
| Sentiment Analysis | NLU | RoBERTa (cardiffnlp) |
| Intent Classification | NLU | BART zero-shot (facebook) |
| Response Generation | NLG | Template-based generator |
| Entity Extraction | NLU | Regex pipeline |

---

## 🚀 Quick Start

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Run API
```bash
uvicorn app.main:app --reload --port 8000
```

### 3. Open Swagger UI
```
http://localhost:8000/docs
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/health` | Model info |
| POST | `/analyze` | Analyze single text |
| POST | `/analyze/batch` | Analyze multiple texts |

---

## 📊 Example

### Input
```json
{
  "text": "I am really unhappy with my order #12345, it arrived late!"
}
```

### Output
```json
{
  "text": "I am really unhappy with my order #12345, it arrived late!",
  "sentiment": {
    "label": "negative",
    "confidence": 0.962,
    "scores": { "negative": 0.962, "neutral": 0.028, "positive": 0.01 }
  },
  "intent": {
    "label": "complaint",
    "confidence": 0.891
  },
  "entities": [
    { "type": "reference_id", "value": "12345" }
  ],
  "nlg_response": "We sincerely apologize for your experience. Our team will look into this immediately.",
  "latency_ms": 312.4
}
```

---

## 🛠 Tech Stack
- **NLU:** HuggingFace Transformers (RoBERTa, BART)
- **NLG:** Template-based response generation
- **API:** FastAPI + Uvicorn
- **Container:** Docker

---

## 🐳 Docker
```bash
docker build -t sentiment-analyzer .
docker run -p 8000:8000 sentiment-analyzer
```
