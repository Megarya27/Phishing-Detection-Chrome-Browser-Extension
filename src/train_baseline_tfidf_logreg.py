# This script trains TF-IDF + logistic regression model for Baseline 4 on same email text
# corpus DistilBERT is trained on so all baselines can be compared on identical data and splits, 
# ensuring fair comparison

import json
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from global_config import MODELS_DIR, RESULTS_DIR, RANDOM_SEED
from data_text import build_text_dataset
from metrics import full_report
MAX_FEATURES = 20000  #max vocabulary size for TF-IDF vectorizer
                       #reduces memory usage also speeds up training

def train_tfidf_logreg(splits, seed=RANDOM_SEED):
    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=(1, 2),  #unigrams and bigrams helpful for capturing phrases like 'verify now'
        min_df=2)  #drop terms that only appear once, mostly noise and typos
    X_train = vectorizer.fit_transform(splits["train"]["text"])
    model = LogisticRegression(max_iter=1000, random_state=seed)
    model.fit(X_train, splits["train"]["label"])
    return vectorizer, model

def evaluate(vectorizer, model, texts, labels):
    X = vectorizer.transform(texts)
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    return y_pred, y_prob

def main():
    splits = build_text_dataset()  # same train/val/test split and cleaning as DistilBERT
    vectorizer, model = train_tfidf_logreg(splits)
    y_pred, y_prob = evaluate(vectorizer, model, splits["test"]["text"], splits["test"]["label"])
    report, f1_dist = full_report("tfidf_logreg_semantic", splits["test"]["label"], y_pred, y_prob)
    print(json.dumps(report, indent=2))
    joblib.dump({"vectorizer": vectorizer, "model": model}, MODELS_DIR / "baseline_tfidf_logreg.joblib")
    np.save(RESULTS_DIR / "baseline_tfidf_logreg_f1_bootstrap.npy", f1_dist)
    with open(RESULTS_DIR / "baseline_tfidf_logreg_report.json", "w") as f:
        json.dump(report, f, indent=2)

if __name__ == "__main__":
    main()
