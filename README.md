# Email_classifier
AI Powered Smart Email Classifier for Enterprises.

## Project Overview
This project classifies emails into complaint, request, feedback, and spam, while predicting their urgency (high, medium, low) to help support teams triage messages.

## Project Structure
* **raw_datasets/** – original CSV datasets.
* **cleaned_datasets/** – cleaned datasets and all_emails_merged.csv.
* **python files/** – scripts for cleaning/merging datasets and running predictions.
* **milestone 3/** – urgency training/testing scripts.
* **LICENSE** – MIT license.
* **README.md** – project documentation.

---

## 📍 Milestone 1: Data Preprocessing & Dataset Creation
**Objective:** Prepare a unified labeled dataset from multiple raw email datasets for downstream ML tasks.

**Work Completed:**
* Converted multiple datasets into a common schema:
    * `text`: Original email text.
    * `cleaned_text`: Cleaned version produced by preprocessing.
    * `category`: One of: complaint, request, feedback, spam.
    * `label`: Integer encoding of category.
    * `urgency`: High, medium, or low.
* Removed duplicates and cleaned invalid rows where applicable.
* Created the final merged labeled dataset: `cleaned_datasets/all_emails_merged.csv`.

**Output Artifact:**
* `cleaned_datasets/all_emails_merged.csv`

## 📍 Milestone 2: Email Category & Spam Prediction
**Objective:** Train and evaluate ML models to classify emails into categories and identify spam.

**Models Trained:**
* TF‑IDF + Logistic Regression
* TF‑IDF + Naive Bayes

**Evaluation Results:**
#### Logistic Regression (Accuracy: 0.9741)
| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| 0 (Complaint) | 0.99 | 0.97 | 0.98 | 3531 |
| 1 (Request) | 0.97 | 0.99 | 0.98 | 4358 |
| 2 (Feedback) | 0.95 | 0.97 | 0.96 | 2396 |
| 3 (Spam) | 0.99 | 0.91 | 0.95 | 991 |

#### Naive Bayes (Accuracy: 0.9501)
* **Note:** Naive Bayes showed a 0.00 F1-score for the Spam category (label 3).

**Demo Prediction:**
* Running `python files/test_model.py` provides predictions using an ensemble of Logistic Regression and DistilBERT.

## 📍 Milestone 3: Urgency Detection & Scoring
**Objective:** Predict email urgency and generate a final score using a hybrid ML and rule-based approach.

**Work Completed:**
* Trained a multiclass urgency model using TF‑IDF + Logistic Regression.
* Built keyword-based urgency probabilities combined with ML probabilities (hybrid).
* Evaluated the classifier using a confusion matrix and classification report.

**Evaluation Results:**
* **Accuracy:** 0.86
* **Weighted F1:** 0.8655
* **Macro F1:** 0.8032

**Confusion Matrix:**
```text
[[7542  368  437]  # Actual: Low
 [ 129 3025  813]  # Actual: Medium
 [  47  211 1541]] # Actual: High
```

### Classification Report
| Urgency | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Low** | 0.98 | 0.90 | 0.94 | 8347 |
| **Medium** | 0.84 | 0.76 | 0.80 | 3967 |
| **High** | 0.55 | 0.86 | 0.67 | 1799 |

### Files Involved
* `milestone 3/model_milestone3_train.py` – trains urgency model and saves pipeline.
* `milestone 3/test_model_milestone3.py` – tests urgency predictions and prints score.
