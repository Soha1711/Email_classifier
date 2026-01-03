import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, f1_score

DATA_PATH = "cleaned_datasets/all_emails_merged.csv"  # <-- your file
TEXT_COL = "text"
URGENCY_COL = "urgency"  # values: low/medium/high

LABEL_MAP = {"low": 0, "medium": 1, "high": 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

def main():
    # 1) Load data
    df = pd.read_csv(DATA_PATH)

    # 2) Keep only required columns + clean nulls
    df = df[[TEXT_COL, URGENCY_COL]].dropna()
    df[TEXT_COL] = df[TEXT_COL].astype(str)
    df[URGENCY_COL] = df[URGENCY_COL].astype(str).str.lower().str.strip()

    # 3) Filter invalid labels (safety)
    df = df[df[URGENCY_COL].isin(LABEL_MAP.keys())].copy()

    # 4) Encode labels
    y = df[URGENCY_COL].map(LABEL_MAP).values
    X = df[TEXT_COL].values

    print("Rows used:", len(df))
    print("Urgency distribution:\n", df[URGENCY_COL].value_counts())

    # 5) Stratified split (keeps label ratio same in train/test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # 6) Pipeline (prevents leakage + keeps vectorizer + model together)
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=50000,        # good for 56k rows
            ngram_range=(1, 2),        # improves urgency phrases like "not working"
            stop_words="english"
        )),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",   # helps if high urgency is less
            n_jobs=-1
        ))
    ])

    # Train
    pipe.fit(X_train, y_train)

    # Predict
    y_pred = pipe.predict(X_test)

    # 7) Evaluation (confusion matrix + F1 + report)
    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    print("\n=== Classification Report ===")
    print(classification_report(
        y_test, y_pred,
        target_names=["low", "medium", "high"]
    ))

    f1_w = f1_score(y_test, y_pred, average="weighted")
    f1_m = f1_score(y_test, y_pred, average="macro")
    print(f"F1 weighted: {f1_w:.4f}")
    print(f"F1 macro   : {f1_m:.4f}")

    # 8) Save trained pipeline
    joblib.dump(pipe, "urgency_pipeline.pkl")
    print("\n✅ Saved model: urgency_pipeline.pkl")

if __name__ == "__main__":
    main()
