import os
import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

RAW_DIR = "dataset1"          # folder where aa_dataset‑tickets CSV is
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

def map_type_to_category(ticket_type, queue):
    t = str(ticket_type).lower()
    q = str(queue).lower()
    if "spam" in t:
        return "spam"
    if "billing" in q or "payment" in q:
        return "complaint"
    if "request" in t:
        return "request"
    if "incident" in t or "problem" in t:
        return "complaint"
    return "feedback"

CATEGORY2ID = {"complaint": 0, "request": 1, "feedback": 2, "spam": 3}

def map_priority_to_urgency(p):
    p = str(p).lower()
    if "high" in p:
        return "high"
    if "medium" in p:
        return "medium"
    if "low" in p:
        return "low"
    return "medium"

def clean_aa_tickets():
    in_path = os.path.join(RAW_DIR, "aa_dataset-tickets-multi-lang-5-2-50-version.csv")
    df = pd.read_csv(in_path)

    # 1) keep only English tickets
    df = df[df["language"] == "en"]

    # 2) drop duplicate tickets based on subject + body
    df["text_raw"] = (df["subject"].fillna("") + " " + df["body"].fillna("")).str.strip()
    df = df.drop_duplicates(subset=["text_raw"])

    # 3) original text column
    df["text"] = df["text_raw"]

    # 4) cleaned text
    df["cleaned_text"] = df["text"].apply(clean_email)

    # 5) category + numeric label
    df["category"] = df.apply(
        lambda r: map_type_to_category(r.get("type", ""), r.get("queue", "")),
        axis=1
    )
    df["label"] = df["category"].map(CATEGORY2ID)

    # 6) urgency from priority
    df["urgency"] = df["priority"].apply(map_priority_to_urgency)

    # 7) keep only requested columns
    out = df[["text", "cleaned_text", "category", "label", "urgency"]]

    out_path = os.path.join(CLEAN_DIR, "aa_dataset-tickets_clean.csv")
    out.to_csv(out_path, index=False)
    print("Saved:", out_path)

if __name__ == "__main__":
    clean_aa_tickets()
