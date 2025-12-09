import os
import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

RAW_DIR = "dataset1"          # folder where fraud_email.csv is
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

CATEGORY2ID = {"complaint": 0, "request": 1, "feedback": 2, "spam": 3}

def map_class_to_category(cls):
    # 1 = fraud / spam, 0 = normal mail
    return "spam" if int(cls) == 1 else "feedback"

def map_text_to_urgency(text):
    t = str(text).lower()
    high_words = [
        "fraud", "unauthorized", "suspicious", "scam", "phishing",
        "identity theft", "account compromised", "stolen", "urgent transfer"
    ]
    if any(w in t for w in high_words):
        return "high"
    # most marketing / generic spam is low urgency
    return "low"

def clean_fraud_email():
    in_path = os.path.join(RAW_DIR, "fraud_email.csv")
    df = pd.read_csv(in_path)

    # 1) original text
    df["text_raw"] = df["Text"].fillna("").astype(str).str.strip()

    # 2) drop duplicate emails
    df = df.drop_duplicates(subset=["text_raw"])

    # 3) unified text
    df["text"] = df["text_raw"]

    # 4) cleaned_text
    df["cleaned_text"] = df["text"].apply(clean_email)

    # 5) category and numeric label from Class
    df["category"] = df["Class"].apply(map_class_to_category)
    df["label"] = df["category"].map(CATEGORY2ID)

    # 6) urgency from email content
    df["urgency"] = df["text"].apply(map_text_to_urgency)

    # 7) keep only required columns
    out = df[["text", "cleaned_text", "category", "label", "urgency"]]

    out_path = os.path.join(CLEAN_DIR, "fraud_email_clean.csv")
    out.to_csv(out_path, index=False)
    print("Saved:", out_path)

if __name__ == "__main__":
    clean_fraud_email()
