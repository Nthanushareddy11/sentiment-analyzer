"""
NLP Pipeline — Sentiment + Intent + NLG
Lightweight rule-based approach — no heavy ML dependencies needed.
"""

import re
from collections import Counter

# ── Word lists ────────────────────────────────────────────────────────
POSITIVE_WORDS = {
    "good","great","excellent","amazing","wonderful","fantastic","love","happy",
    "pleased","satisfied","awesome","best","perfect","thank","thanks","appreciate",
    "helpful","easy","fast","quick","friendly","recommend","impressed","beautiful",
    "brilliant","superb","outstanding","exceptional","delighted","enjoy","enjoyed"
}

NEGATIVE_WORDS = {
    "bad","terrible","awful","horrible","hate","disappointed","disappointing","poor",
    "worst","useless","broken","failed","wrong","problem","issue","error","slow",
    "rude","unhappy","angry","frustrated","annoyed","upset","disgusted","refund",
    "cancel","damaged","late","missing","never","complaint","complain","unacceptable"
}

NEUTRAL_WORDS = {
    "okay","ok","fine","average","normal","standard","usual","typical","moderate"
}

INTENT_PATTERNS = {
    "question"    : [r"\?", r"\bwhat\b", r"\bhow\b", r"\bwhen\b", r"\bwhere\b", r"\bwhy\b", r"\bwho\b", r"\bcan you\b", r"\bdo you\b"],
    "complaint"   : [r"\bcomplaint\b", r"\bcomplain\b", r"\bunhappy\b", r"\bdisappointed\b", r"\bterrible\b", r"\bawful\b", r"\bbroken\b", r"\bdamaged\b", r"\bnot working\b"],
    "cancellation": [r"\bcancel\b", r"\bcancellation\b", r"\bunsubscribe\b", r"\bstop\b", r"\bend my\b", r"\bterminate\b"],
    "request"     : [r"\bplease\b", r"\bcould you\b", r"\bwould you\b", r"\bi need\b", r"\bi want\b", r"\bi'd like\b", r"\brequest\b"],
    "purchase"    : [r"\bbuy\b", r"\border\b", r"\bpurchase\b", r"\bpayment\b", r"\bprice\b", r"\bcost\b", r"\bcheckout\b"],
    "feedback"    : [r"\bfeedback\b", r"\breview\b", r"\bsuggestion\b", r"\bopinion\b", r"\bthought\b", r"\brating\b"],
    "greeting"    : [r"\bhello\b", r"\bhi\b", r"\bhey\b", r"\bgood morning\b", r"\bgood afternoon\b", r"\bgreetings\b"],
}

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
    """No heavy models needed — rule-based approach."""
    print("✅ Lightweight NLP pipeline ready!")
    return {"ready": True}


def extract_entities(text: str) -> list:
    entities = []
    emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    for e in emails:
        entities.append({"type": "email", "value": e})
    prices = re.findall(r'\$[\d,]+(?:\.\d{2})?|\b\d+(?:\.\d{2})?\s*(?:dollars?|USD|INR|rupees?)\b', text, re.IGNORECASE)
    for p in prices:
        entities.append({"type": "amount", "value": p})
    dates = re.findall(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}(?:,\s*\d{4})?\b|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', text, re.IGNORECASE)
    for d in dates:
        entities.append({"type": "date", "value": d})
    orders = re.findall(r'\b(?:order|ticket|case|ref|id)[#:\s]+([A-Z0-9-]+)\b', text, re.IGNORECASE)
    for o in orders:
        entities.append({"type": "reference_id", "value": o})
    return entities


def get_sentiment(models, text: str) -> dict:
    words = re.findall(r'\b\w+\b', text.lower())
    pos = sum(1 for w in words if w in POSITIVE_WORDS)
    neg = sum(1 for w in words if w in NEGATIVE_WORDS)
    neu = sum(1 for w in words if w in NEUTRAL_WORDS)
    total = pos + neg + neu + 1

    if pos > neg:
        label = "positive"
        confidence = round(min(0.95, 0.55 + (pos - neg) / total), 3)
    elif neg > pos:
        label = "negative"
        confidence = round(min(0.95, 0.55 + (neg - pos) / total), 3)
    else:
        label = "neutral"
        confidence = round(0.55 + neu / total * 0.3, 3)

    pos_score = round(pos / total, 3)
    neg_score = round(neg / total, 3)
    neu_score = round(1 - pos_score - neg_score, 3)

    return {
        "label"     : label,
        "confidence": confidence,
        "scores"    : {"positive": pos_score, "neutral": max(0, neu_score), "negative": neg_score}
    }


def get_intent(models, text: str) -> dict:
    text_lower = text.lower()
    scores = {}
    for intent, patterns in INTENT_PATTERNS.items():
        score = sum(1 for p in patterns if re.search(p, text_lower))
        scores[intent] = round(score / len(patterns), 3)

    if not any(scores.values()):
        scores["other"] = 1.0
        best_intent = "other"
    else:
        best_intent = max(scores, key=scores.get)
        if scores[best_intent] == 0:
            best_intent = "other"
            scores["other"] = 0.5

    sorted_scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
    return {
        "label"     : best_intent,
        "confidence": round(scores.get(best_intent, 0.5), 3),
        "all_scores": sorted_scores
    }


def generate_response(sentiment_label: str, intent_label: str) -> str:
    key = (sentiment_label, intent_label)
    if key in NLG_TEMPLATES:
        return NLG_TEMPLATES[key]
    fallbacks = {
        "positive": "Thank you for your message! We're happy to assist.",
        "negative": "We apologize for any inconvenience. We're here to help.",
        "neutral" : "Thank you for contacting us. We'll respond shortly.",
    }
    return fallbacks.get(sentiment_label, "Thank you for reaching out!")


def analyze_text(models: dict, text: str) -> dict:
    sentiment = get_sentiment(models, text)
    intent    = get_intent(models, text)
    entities  = extract_entities(text)
    response  = generate_response(sentiment["label"], intent["label"])
    return {
        "sentiment"    : sentiment,
        "intent"       : intent,
        "entities"     : entities,
        "nlg_response" : response,
    }
