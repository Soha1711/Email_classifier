import os
import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

RAW_DIR = "dataset1"          # folder where dataset1.csv is
CLEAN_DIR = "datasets_clean"
os.makedirs(CLEAN_DIR, exist_ok=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_email(text):
    text = str(text).lower()
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

# map (dataset1) intent/category to complaint/request/feedback/spam
def map_intent_to_category(intent, cat):
    i = str(intent).lower()
    c = str(cat).lower()
    if "spam" in i or "spam" in c:
        return "spam"
    if "cancel" in i or "refund" in i or "order" in c:
        return "complaint"
    if "request" in i or "question" in i:
        return "request"
    if "feedback" in i or "praise" in i:
        return "feedback"
    return "request"

CATEGORY2ID = {"complaint": 0, "request": 1, "feedback": 2, "spam": 3}

def map_text_to_urgency(text):
    t = str(text).lower()
    high_words = ["urgent", "immediately", "asap", "not working",
                  "error", "failed", "cant pay", "cannot pay"]
    medium_words = ["cancel order", "cancel purchase", "refund",
                    "problem", "issue", "help", "support"]

    if any(w in t for w in high_words):
        return "high"
    if any(w in t for w in medium_words):
        return "medium"
    return "low"

def clean_dataset1():
    in_path = os.path.join(RAW_DIR, "dataset1.csv")
    df = pd.read_csv(in_path)

    # 1) original text = customer's instruction
    df["text_raw"] = df["instruction"].fillna("").astype(str).str.strip()

    # 2) drop duplicate texts
    df = df.drop_duplicates(subset=["text_raw"])

    # 3) use text_raw as text
    df["text"] = df["text_raw"]

    # 4) cleaned text
    df["cleaned_text"] = df["text"].apply(clean_email)

    # 5) category and numeric label
    df["category"] = df.apply(
        lambda r: map_intent_to_category(r.get("intent", ""), r.get("category", "")),
        axis=1
    )
    df["label"] = df["category"].map(CATEGORY2ID)

    # 6) urgency from text content
    df["urgency"] = df["text"].apply(map_text_to_urgency)

    # 7) keep only requested columns
    out = df[["text", "cleaned_text", "category", "label", "urgency"]]

    out_path = os.path.join(CLEAN_DIR, "dataset1_clean.csv")
    out.to_csv(out_path, index=False)
    print("Saved:", out_path)

if __name__ == "__main__":
    clean_dataset1()
