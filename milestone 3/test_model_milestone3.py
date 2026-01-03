import joblib
import numpy as np

pipe = joblib.load("urgency_pipeline.pkl")

# Rule keywords (edit/expand based on your dataset)
HIGH = ["urgent", "asap", "immediately", "not working", "down", "failed", "blocked", "can't access"]
MED  = ["please", "request", "help", "issue", "refund", "status", "query"]
LOW  = ["thank", "thanks", "appreciate", "feedback", "unsubscribe"]

LABELS = ["low", "medium", "high"]

def rule_probs(text: str):
    t = text.lower()
    scores = np.array([
        sum(k in t for k in LOW),
        sum(k in t for k in MED),
        sum(k in t for k in HIGH),
    ], dtype=float)

    # smoothing so it never becomes [0,0,0]
    scores = scores + 0.2
    return scores / scores.sum()

def hybrid_predict(text: str, alpha: float = 0.7):
    # ML probabilities from pipeline
    ml_p = pipe.predict_proba([text])[0]

    # Rule probabilities
    r_p = rule_probs(text)

    # Weighted combination (hybrid)
    final_p = alpha * ml_p + (1 - alpha) * r_p

    label = LABELS[int(np.argmax(final_p))]
    urgency_score_high = float(final_p[2]) * 100  # 0-100

    return {
        "label": label,
        "score_high": round(urgency_score_high, 2),
        "ml_probs": [round(float(x), 4) for x in ml_p],
        "rule_probs": [round(float(x), 4) for x in r_p],
        "final_probs": [round(float(x), 4) for x in final_p],
    }

tests = [
    "Internet not working please fix asap",
    "Need help with refund status",
    "Thank you for the quick update",
]

for t in tests:
    print("\nTEXT:", t)
    print(hybrid_predict(t))
