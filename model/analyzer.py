"""
NLP Pipeline — Sentiment + Intent + NLG
----------------------------------------
- Sentiment  : HuggingFace RoBERTa (NLU)
- Intent     : Zero-shot classification via BART (NLU)
- NLG        : Template-based response generator (NLG)
- Entities   : Simple keyword extraction
"""

from transformers import pipeline
import re


# ── Intent labels ────────────────────────────────────────────────────
INTENT_LABELS = [
    "question",
    "complaint",
    "request",
    "feedback",
    "greeting",
    "cancellation",
    "purchase",
    "other"
]

# ── NLG Response templates ────────────────────────────────────────────
NLG_TEMPLATES = {
    ("positive", "feedback")     : "Thank you for your positive feedback! We're thrilled you had a great experience.",
    ("positive", "purchase")     : "Great choice! We hope you enjoy your purchase. Feel free to reach out if you need help.",
    ("positive", "greeting")     : "Hello! Great to hear from you. How can I assist you today?",
    ("positive", "other")        : "Thank you for reaching out! We're glad to hear things are going well.",
    ("negative", "complaint")    : "We sincerely apologize for your experience. Our team will look into this immediately.",
    ("negative", "cancellation") : "We're sorry to see you go. If there's anything we can do to improve, please let us know.",
    ("negative", "other")        : "We're sorry to hear that. Please let us know how we can make this right for you.",
    ("neutral",  "question")     : "Great question! Let me help you find the information you need.",
    ("neutral",  "request")      : "We've received your request and will process it as soon as possible.",
    ("neutral",  "other")        : "Thank you for contacting us. We'll get back to you shortly.",
}


def load_models():
    """Load sentiment and zero-shot classification models."""
    print("Loading sentiment model (RoBERTa)...")
    sentiment_model = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        return_all_scores=True,
    )

    print("Loading intent classifier (BART zero-shot)...")
    intent_model = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
    )

    return {
        "sentiment" : sentiment_model,
        "intent"    : intent_model,
    }


def extract_entities(text: str) -> list:
    """Simple keyword/entity extraction using regex patterns."""
    entities = []

    # Extract emails
    emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    for e in emails:
        entities.append({"type": "email", "value": e})

    # Extract prices/numbers
    prices = re.findall(r'\$[\d,]+(?:\.\d{2})?|\b\d+(?:\.\d{2})?\s*(?:dollars?|USD|INR|rupees?)\b', text, re.IGNORECASE)
    for p in prices:
        entities.append({"type": "amount", "value": p})

    # Extract dates
    dates = re.findall(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}(?:,\s*\d{4})?\b|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', text, re.IGNORECASE)
    for d in dates:
        entities.append({"type": "date", "value": d})

    # Extract order/ticket numbers
    orders = re.findall(r'\b(?:order|ticket|case|ref|id)[#:\s]+([A-Z0-9-]+)\b', text, re.IGNORECASE)
    for o in orders:
        entities.append({"type": "reference_id", "value": o})

    return entities


def get_sentiment(model, text: str) -> dict:
    """Run sentiment analysis and return label + confidence."""
    results = model(text[:512])[0]   # RoBERTa max 512 tokens

    # Map labels
    label_map = {
        "positive": "positive",
        "negative": "negative",
        "neutral" : "neutral",
        "LABEL_0" : "negative",
        "LABEL_1" : "neutral",
        "LABEL_2" : "positive",
    }

    best = max(results, key=lambda x: x["score"])
    label = label_map.get(best["label"].lower(), best["label"].lower())

    return {
        "label"     : label,
        "confidence": round(best["score"], 3),
        "scores"    : {
            label_map.get(r["label"].lower(), r["label"].lower()): round(r["score"], 3)
            for r in results
        }
    }


def get_intent(model, text: str) -> dict:
    """Run zero-shot intent classification."""
    result = model(text[:512], candidate_labels=INTENT_LABELS)

    return {
        "label"     : result["labels"][0],
        "confidence": round(result["scores"][0], 3),
        "all_scores": {
            label: round(score, 3)
            for label, score in zip(result["labels"], result["scores"])
        }
    }


def generate_response(sentiment_label: str, intent_label: str) -> str:
    """Generate NLG response based on sentiment + intent."""
    key = (sentiment_label, intent_label)
    if key in NLG_TEMPLATES:
        return NLG_TEMPLATES[key]

    # Fallback by sentiment only
    fallbacks = {
        "positive": "Thank you for your message! We're happy to assist.",
        "negative": "We apologize for any inconvenience. We're here to help.",
        "neutral" : "Thank you for contacting us. We'll respond shortly.",
    }
    return fallbacks.get(sentiment_label, "Thank you for reaching out!")


def analyze_text(models: dict, text: str) -> dict:
    """Full NLP pipeline: sentiment + intent + entities + NLG."""
    sentiment = get_sentiment(models["sentiment"], text)
    intent    = get_intent(models["intent"], text)
    entities  = extract_entities(text)
    response  = generate_response(sentiment["label"], intent["label"])

    return {
        "sentiment"    : sentiment,
        "intent"       : intent,
        "entities"     : entities,
        "nlg_response" : response,
    }
