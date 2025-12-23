import torch
import torch.nn.functional as F
import joblib
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# same mapping as training
id2label = {
    0: "complaint",
    1: "request",
    2: "feedback",
    3: "spam",
}

# ----- load DistilBERT -----
model_path = "./distilbert_email_model"
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
bert_model = DistilBertForSequenceClassification.from_pretrained(model_path)

# ----- load Logistic Regression + TF‑IDF -----
tfidf = joblib.load("tfidf_vectorizer.joblib")
logreg = joblib.load("logreg_email_model.joblib")

# ----- emails to test -----
test_emails = [
    "Internet not working, please fix this issue as soon as possible.",
    "Congratulations! You have won a free iPhone. Click here to claim now.",
    "I want to know the status of my refund request",
    "Your service is terrible, I am very disappointed",
    "Great product, I really liked the new update",
    "i request a call back regarding my account issues",
    "you won an iphone for free. congratulations."
]

def predict_lr(email_text: str) -> int:
    X = tfidf.transform([email_text])
    return int(logreg.predict(X)[0])

def predict_bert(email_text: str) -> int:
    inputs = tokenizer(
        email_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )
    with torch.no_grad():
        outputs = bert_model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        pred_id = int(torch.argmax(probs, dim=1).item())
        spam_prob = float(probs[0, 3].item())
    return pred_id, spam_prob

bert_model.eval()
for email in test_emails:
    lr_pred = predict_lr(email)
    bert_pred, spam_prob = predict_bert(email)

    # FINAL decision = Logistic Regression prediction
    final_pred = lr_pred

    print("\nEmail:", email)
    print("LR prediction:", id2label[lr_pred])
    print(f"DistilBERT prediction: {id2label[bert_pred]} (spam_prob={spam_prob:.3f})")
    print("Final prediction (used):", id2label[final_pred])
