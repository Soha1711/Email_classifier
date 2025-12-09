import os
import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

RAW_DIR = "dataset1"                 # folder where spam-detection-email.csv is
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

def map_cat_to_category(cat):
    c = str(cat).lower()
    if c == "spam":
        return "spam"
    # treat ham as normal user messages (feedback / general)
    return "feedback"

def map_text_to_urgency(text):
    t = str(text).lower()
    high_words = ["urgent", "immediately", "asap", "not working", "error", "failed"]
    medium_words = ["help", "problem", "issue", "call me", "need", "please call"]

    if any(w in t for w in high_words):
        return "high"
    if any(w in t for w in medium_words):
        return "medium"
    return "low"

def clean_spam_detection():
    in_path = os.path.join(RAW_DIR, "spam detection email.csv")
    df = pd.read_csv(in_path)

    # 1) original text
    df["text_raw"] = df["Message"].fillna("").astype(str).str.strip()

    # 2) drop duplicate messages
    df = df.drop_duplicates(subset=["text_raw"])

    # 3) unified text
    df["text"] = df["text_raw"]

    # 4) cleaned_text
    df["cleaned_text"] = df["text"].apply(clean_email)

    # 5) category + numeric label
    df["category"] = df["Category"].apply(map_cat_to_category)
    df["label"] = df["category"].map(CATEGORY2ID)

    # 6) urgency from content
    df["urgency"] = df["text"].apply(map_text_to_urgency)

    # 7) keep only requested columns
    out = df[["text", "cleaned_text", "category", "label", "urgency"]]

    out_path = os.path.join(CLEAN_DIR, "spam-detection-email_clean.csv")
    out.to_csv(out_path, index=False)
    print("Saved:", out_path)

if __name__ == "__main__":
    clean_spam_detection()
