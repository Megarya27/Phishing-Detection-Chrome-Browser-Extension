# This program builds teh NLP training corpus for DistilBERT
# using the Nazario Phishing Email Corpus which contains only phishing Emails
# and the SpamAssassin Public Corpus which contains mostly legitamate emails.
# Additionally, datasets like Enron / Ling / Nigerian_Fraud / CEAS_08 will  be used
# in case if additional data is required for further model improvement. 

import re
from html import unescape

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

from global_config import (
    NAZARIO_CSV, SPAMASSASSIN_CSV, ENRON_CSV, LING_CSV, NIGERIAN_FRAUD_CSV,
    CEAS_CSV, MIN_TEXT_TRAINING_SAMPLES, RANDOM_SEED,
)

#high limit set for reading csv safely
CSV_FIELD_SIZE_LIMIT = 10_000_000
#regex patterns for scrubbing PII from email text
EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
PHONE_RE = re.compile(r"(\+?\d{1,3}[\s.-]?)?(\(?\d{2,4}\)?[\s.-]?){2,4}\d{3,4}")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
HTML_TAG_RE = re.compile(r"<[^>]+>")
QUOTED_REPLY_RE = re.compile(r"^(>.*|On .* wrote:)$", re.MULTILINE)
WHITESPACE_RE = re.compile(r"\s+")
NAME_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b")

# removes HTML tags and decodes HTML entities to plain text
def strip_html(text: str) -> str:
    return HTML_TAG_RE.sub(" ", unescape(text))

#replaces sensitive info with generic placeholders
def scrub_pii(text: str) -> str:
    text = EMAIL_RE.sub("[EMAIL]", text)
    text = URL_RE.sub("[URL]", text)
    text = PHONE_RE.sub("[PHONE]", text)
    text = NAME_RE.sub("[NAME]", text)
    return text

#combines subject and body, cleans the text, and returns a single string
def clean_email_text(subject: str, body: str) -> str:
    # Handles missing fields
    if subject is None:
        subject = ""
    if body is None:
        body = ""

    text = f"{subject}\n{body}"
    text = strip_html(text)
    text = QUOTED_REPLY_RE.sub(" ", text)
    text = scrub_pii(text)
    #ensure text is valid UTF-8, dropping unencodable characters
    text = text.encode("utf-8", errors="ignore").decode("utf-8")
    text = WHITESPACE_RE.sub(" ", text).strip() #replace multiple whitespace with single space and trim
    return text

#safely load csv file containing email data
def _load_csv(path, label_col="label"):
    import csv
    csv.field_size_limit(CSV_FIELD_SIZE_LIMIT)
    
    df = pd.read_csv(path, engine="python", on_bad_lines="skip",encoding="utf-8", encoding_errors="ignore")
    
    #convert labels to numbers, replacing invalid strings with NaN, then drop NaNs
    df[label_col] = pd.to_numeric(df[label_col], errors="coerce")
    df = df.dropna(subset=[label_col])
    
    #convert label column to whole numbers for consistency across datasets
    df[label_col] = df[label_col].astype(int)
    return df

#ensure every dataset uses same column names
def _standardise(df, source_name):
    # If CSV doesnt have subject/body column, create empty ones
    if "subject" not in df.columns:
        df["subject"] = ""
    if "body" not in df.columns:
        df["body"] = ""
        
    #keep only necessary columns and add source column for tracking
    out = df[["subject", "body", "label"]].copy()
    out["source"] = source_name
    return out

#load Nazario and SpamAssassin, also add addtional datasets if needed
def load_raw_corpora():
    nazario_df = _standardise(_load_csv(NAZARIO_CSV), "nazario")
    spamassassin_df = _standardise(_load_csv(SPAMASSASSIN_CSV), "spamassassin")
    combined = pd.concat([nazario_df, spamassassin_df], ignore_index=True) #combines both datasets into single table
    #additional datasets if needed
    supplements = [
        (ENRON_CSV, "enron"),
        (LING_CSV, "ling"),
        (NIGERIAN_FRAUD_CSV, "nigerian_fraud"),
        (CEAS_CSV, "ceas_08")]
    
    for file_path, source_name in supplements:
        # Stop adding if we have 3x minimum
        if len(combined) >= (MIN_TEXT_TRAINING_SAMPLES * 3):
            break
            
        if file_path.exists():
            extra_df = _standardise(_load_csv(file_path), source_name)
            combined = pd.concat([combined, extra_df], ignore_index=True)
            
    return combined

#prep final dataset by loading raw text, cleaning, split into train/val/test, and balancing the training set
def build_text_dataset(seed=RANDOM_SEED):
    df = load_raw_corpora()
    
    #combine subject + body into single cleaned 'text' column 
    cleaned_texts = []
    for subject, body in zip(df["subject"], df["body"]):
        cleaned_texts.append(clean_email_text(subject, body))
    df["text"] = cleaned_texts

    # drop emails under 20 characters an remove dupes
    df = df[df["text"].str.len() > 20]
    df = df.drop_duplicates(subset=["text"])
    df = df.reset_index(drop=True)

    #throw error if not enough samples for training
    if len(df) < MIN_TEXT_TRAINING_SAMPLES:
        raise ValueError(
            f"Only {len(df)} usable text samples found; we need "
            f">= {MIN_TEXT_TRAINING_SAMPLES}. Please add more source files."
        )

    X = df[["text", "source"]]
    y = df["label"]

    # split into 70% Training data and 30% temporary data for validation/test
    X_train, X_val_test, y_train, y_val_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=seed
    )
    
    # Split 30% temporary data into 15% Validation and 15% Test sets
    X_val, X_test, y_val, y_test = train_test_split(
        X_val_test, y_val_test, test_size=0.50, stratify=y_val_test, random_state=seed
    )

    #recombine training features and labels temporarily to balance them
    train = pd.concat([X_train, y_train], axis=1)
    
    #Find out which label (0 or 1) has fewer samples as it is the minority class
    minority_label = train["label"].value_counts().idxmin()
    majority_label = 1 - minority_label
    
    minority = train[train["label"] == minority_label]
    majority = train[train["label"] == majority_label]
    
    # remove samples from majority class to match count of minority class for 50/50 balance
    majority_down = resample(
        majority, 
        replace=False, 
        n_samples=len(minority),
        random_state=seed
    )
    
    #combine back the minority and downsampled majority class, shuffle, and reset index
    train_balanced = pd.concat([minority, majority_down])
    train_balanced = train_balanced.sample(frac=1.0, random_state=seed) #shuffle the rows randomly
    train_balanced = train_balanced.reset_index(drop=True) # reset index numbers

    # return all 3 dataset splits in dictionary
    return {
        "train": train_balanced,
        "val": pd.concat([X_val, y_val], axis=1).reset_index(drop=True),
        "test": pd.concat([X_test, y_test], axis=1).reset_index(drop=True),
    }


if __name__ == "__main__":
    splits = build_text_dataset()
    for name, df in splits.items():
        print(name, len(df), "label balance:", df["label"].value_counts().to_dict())
