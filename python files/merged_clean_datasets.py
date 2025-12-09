import os
import glob
import csv
import pandas as pd

CLEAN_DIR = "datasets_clean"          # folder containing all *_clean.csv
OUTPUT_FILE = "all_emails_merged.csv"

def merge_cleaned_datasets():
    pattern = os.path.join(CLEAN_DIR, "*_clean.csv")
    files = glob.glob(pattern)
    print("Cleaned files found:", files)

    dfs = []
    for path in files:
        df = pd.read_csv(path)
        # keep only standard columns
        df = df[["text", "cleaned_text", "category", "label", "urgency"]]
        # remove newline characters inside text columns
        df["text"] = df["text"].astype(str).str.replace("\n", " ", regex=False)
        df["cleaned_text"] = df["cleaned_text"].astype(str).str.replace("\n", " ", regex=False)
        dfs.append(df)

    all_data = pd.concat(dfs, ignore_index=True)
    print("Merged shape:", all_data.shape)

    out_path = os.path.join(CLEAN_DIR, OUTPUT_FILE)
    all_data.to_csv(
        out_path,
        index=False,
        quoting=csv.QUOTE_ALL,   # quote all fields to keep commas inside cells
        escapechar="\\"
    )
    print("Saved merged file ->", out_path)

if __name__ == "__main__":
    merge_cleaned_datasets()
