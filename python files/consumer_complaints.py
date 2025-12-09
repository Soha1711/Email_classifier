import os
import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

RAW_DIR = "dataset1"          # folder where datatset_consumer_complaints.csv is
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

def map_issue_to_urgency(issue):
    t = str(issue).lower()
    high_words = ["fraud", "identity theft", "illegal action",
                  "foreclosure", "can't repay", "unable to pay"]
    medium_words = ["billing", "fees", "charges", "problems",
                    "investigation", "disputes", "collection"]

    if any(w in t for w in high_words):
        return "high"
    if any(w in t for w in medium_words):
        return "medium"
    return "low"

def clean_consumer_complaints():
    in_path = os.path.join(RAW_DIR, "datatset_consumer_complaints.csv")
    df = pd.read_csv(in_path)

    # 1) text comes from Issue column
    df["text_raw"] = df["Issue"].fillna("").astype(str).str.strip()

    # 2) drop duplicate issues
    df = df.drop_duplicates(subset=["text_raw"])

    # 3) unified text
    df["text"] = df["text_raw"]

    # 4) cleaned_text
    df["cleaned_text"] = df["text"].apply(clean_email)

    # 5) category: all rows are complaints
    df["category"] = "complaint"
    df["label"] = CATEGORY2ID["complaint"]

    # 6) urgency from issue content
    df["urgency"] = df["text"].apply(map_issue_to_urgency)

    # 7) keep only requested columns
    out = df[["text", "cleaned_text", "category", "label", "urgency"]]

    out_path = os.path.join(CLEAN_DIR, "datatset_consumer_complaints_clean.csv")
    out.to_csv(out_path, index=False)
    print("Saved:", out_path)

if __name__ == "__main__":
    clean_consumer_complaints()
