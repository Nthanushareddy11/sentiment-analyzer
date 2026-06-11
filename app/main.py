"""
Sentiment + Intent Analyzer API
--------------------------------
Endpoints:
    GET  /           → health check
    GET  /health     → model info
    POST /analyze    → analyze text (sentiment + intent + NLG response)
    POST /analyze/batch → analyze multiple texts
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.analyzer import load_models, analyze_text

# ── App setup ────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Sentiment + Intent Analyzer API",
    description = "NLP pipeline for real-time sentiment analysis, intent classification and NLG response generation using HuggingFace Transformers.",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# ── Load models at startup ────────────────────────────────────────────
models = None

@app.on_event("startup")
async def startup():
    global models
    print("Loading NLP models...")
    models = load_models()
    print("✅ Models loaded!")


# ── Request/Response schemas ─────────────────────────────────────────
class TextInput(BaseModel):
    text: str
    language: str = "en"

class BatchInput(BaseModel):
    texts: List[str]


# ── Routes ───────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "Sentiment + Intent Analyzer API is running 🟢", "docs": "/docs"}


@app.get("/health")
def health():
    return {
        "status"     : "healthy",
        "models"     : {
            "sentiment" : "cardiffnlp/twitter-roberta-base-sentiment-latest",
            "intent"    : "rule-based + zero-shot",
            "nlg"       : "template-based response generator"
        },
        "tasks"      : ["sentiment analysis (NLU)", "intent classification (NLU)", "response generation (NLG)"],
    }


@app.post("/analyze")
async def analyze(input: TextInput):
    """
    Analyze a single text.

    Returns
    -------
    - sentiment    : positive / negative / neutral + confidence
    - intent       : question / complaint / request / feedback / greeting / other
    - entities     : key topics extracted from text
    - nlg_response : auto-generated response based on sentiment + intent
    - latency_ms   : processing time
    """
    if not input.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    start = time.time()
    result = analyze_text(models, input.text)
    latency = round((time.time() - start) * 1000, 1)

    return JSONResponse({
        "text"         : input.text,
        "sentiment"    : result["sentiment"],
        "intent"       : result["intent"],
        "entities"     : result["entities"],
        "nlg_response" : result["nlg_response"],
        "latency_ms"   : latency,
    })


@app.post("/analyze/batch")
async def analyze_batch(input: BatchInput):
    """Analyze multiple texts at once (max 20)."""
    if len(input.texts) > 20:
        raise HTTPException(status_code=400, detail="Max 20 texts per batch.")

    results = []
    for text in input.texts:
        if text.strip():
            result = analyze_text(models, text)
            results.append({
                "text"         : text,
                "sentiment"    : result["sentiment"],
                "intent"       : result["intent"],
                "nlg_response" : result["nlg_response"],
            })

    return JSONResponse({"results": results, "count": len(results)})
