# Email_classifier

AI Powered Smart Email Classifier for Enterprises.

## Project overview

This project classifies emails into **complaint, request, feedback, spam**  
and predicts their **urgency** (**high, medium, low**) to help support teams triage messages.

## Project structure

- `raw_datasets/` – original CSV datasets.
- `cleaned_datasets/` – cleaned datasets and `all_emails_merged.csv`.
- `python files/` – Python scripts for cleaning and merging datasets.
- `LICENSE` – MIT license.
- `README.md` – project description and usage.

## Data preprocessing

All datasets are converted to a common schema:

- `text` – original email text.
- `cleaned_text` – cleaned version produced by `clean_email`.
- `category` – one of: `complaint`, `request`, `feedback`, `spam`.
- `label` – integer encoding of `category`.
- `urgency` – `high`, `medium`, or `low`.

Duplicates and non‑English rows are removed where applicable.



