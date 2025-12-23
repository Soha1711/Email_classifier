# =========================================================
# Milestone 2: LR + Naive Bayes + DistilBERT (TRAIN)
# =========================================================
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

import torch
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

# ---------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------
df = pd.read_csv(
    r"C:\Users\hp\OneDrive\esfp-2\OneDrive\Desktop\email classifier\cleaned_datasets\all_emails_merged.csv"
)

# Use cleaned_text as input; label is numeric 0,1,2,3
df = df[["cleaned_text", "label"]].dropna().reset_index(drop=True)
df = df.rename(columns={"cleaned_text": "text"})

print("Dataset size:", df.shape[0])
print("Label distribution:\n", df["label"].value_counts())

texts = df["text"].astype(str).tolist()
labels_num = df["label"].astype(int).tolist()

# ---------------------------------------------------------
# 2. Train / test split for baseline models
# ---------------------------------------------------------
X_train_texts, X_test_texts, y_train_num, y_test_num = train_test_split(
    texts, labels_num, test_size=0.2, random_state=42, stratify=labels_num
)

# ---------------------------------------------------------
# 3. TF‑IDF vectorization
# ---------------------------------------------------------
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    stop_words="english",
)

X_train = vectorizer.fit_transform(X_train_texts)
X_test = vectorizer.transform(X_test_texts)

# ---------------------------------------------------------
# 4. Baseline 1: Logistic Regression
# ---------------------------------------------------------
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train_num)

y_pred_lr = logreg.predict(X_test)

print("\n=== Logistic Regression ===")
print(f"Accuracy: {accuracy_score(y_test_num, y_pred_lr):.4f}")
print("Classification report:")
print(classification_report(y_test_num, y_pred_lr))

# ---------------------------------------------------------
# 5. Baseline 2: Naive Bayes
# ---------------------------------------------------------
nb = MultinomialNB()
nb.fit(X_train, y_train_num)

y_pred_nb = nb.predict(X_test)

print("\n=== Naive Bayes ===")
print(f"Accuracy: {accuracy_score(y_test_num, y_pred_nb):.4f}")
print("Classification report:")
print(classification_report(y_test_num, y_pred_nb))

import joblib  # add at top of file

# ... after LR training and printing metrics:

# Save TF‑IDF vectorizer and LR model
joblib.dump(vectorizer, "tfidf_vectorizer.joblib")
joblib.dump(logreg, "logreg_email_model.joblib")


# ---------------------------------------------------------
# ---------------------------------------------------------
# 6. DistilBERT fine‑tuning
# ---------------------------------------------------------
id2label = {
    0: "complaint",
    1: "request",
    2: "feedback",
    3: "spam",
}
label2id = {v: k for k, v in id2label.items()}

texts_train, texts_val, labels_train, labels_val = train_test_split(
    texts, labels_num, test_size=0.2, random_state=42, stratify=labels_num
)

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def encode_texts(text_list):
    return tokenizer(
        text_list,
        truncation=True,
        padding=True,
        max_length=128,
    )

encodings_train = encode_texts(texts_train)
encodings_val = encode_texts(texts_val)

class EmailDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = EmailDataset(encodings_train, labels_train)
val_dataset = EmailDataset(encodings_val, labels_val)

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=4,
    id2label=id2label,
    label2id=label2id,
)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,              # train for a full 1 epoch
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir="./logs",
    logging_steps=200,
    save_strategy="no",
    # <-- no max_steps here
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

print("\n=== DistilBERT fine‑tuning ===")
trainer.train()

model.save_pretrained("./distilbert_email_model")
tokenizer.save_pretrained("./distilbert_email_model")

eval_results = trainer.evaluate()
print("Validation results:", eval_results)



