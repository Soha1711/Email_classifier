from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os

app = FastAPI(title="Email Intelligence API")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

category_model = joblib.load(os.path.join(BASE_DIR, "category_pipeline.pkl"))
urgency_model = joblib.load(os.path.join(BASE_DIR, "urgency_pipeline.pkl"))

# ===== LABEL MAP =====
CATEGORY_MAP = {
    0: "complaint",
    1: "request",
    2: "feedback",
    3: "spam"
}

URGENCY_MAP = {
    0: "low",
    1: "medium",
    2: "high"
}

class EmailRequest(BaseModel):
    sender: str
    subject: str
    text: str


@app.get("/")
def root():
    return {"status": "Backend running"}


@app.post("/predict")
def predict_email(data: EmailRequest):

    text = data.text

    category = category_model.predict([text])[0]
    urgency = urgency_model.predict([text])[0]

    return {
        "email": text,
        "category": str(category),   # already text
        "urgency": str(urgency)     # already text
    }

