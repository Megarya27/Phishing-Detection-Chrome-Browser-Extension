#This script trains SVM baseline (Baseline 2) on same 30 UCI structural features as the
# XGBoost baseline so both can be compared on identical data and splits, ensuring fair comparison

import json
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from global_config import MODELS_DIR, RESULTS_DIR, RANDOM_SEED
from data_structural import load_structural_splits
from metrics import full_report

def train_svm(splits, seed=RANDOM_SEED):
    model = Pipeline([
        ("scaler", StandardScaler()), #rescales features to mean=0 and std=1 for SVM training
        ("svm", SVC(kernel="rbf", probability=True, random_state=seed))])
    model.fit(splits["X_train"], splits["y_train"])
    return model

def evaluate(model, X, y):
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    return y_pred, y_prob

def main():
    splits = load_structural_splits()  #same 30 UCI features used and 70/15/15 split as XGBoost baseline
    model = train_svm(splits)
    y_pred, y_prob = evaluate(model, splits["X_test"], splits["y_test"])
    report, f1_dist = full_report("svm_structural", splits["y_test"], y_pred, y_prob)
    print(json.dumps(report, indent=2))
    joblib.dump(model, MODELS_DIR / "baseline_svm.joblib")
    np.save(RESULTS_DIR / "baseline_svm_f1_bootstrap.npy", f1_dist)
    with open(RESULTS_DIR / "baseline_svm_report.json", "w") as f:
        json.dump(report, f, indent=2)

if __name__ == "__main__":
    main()
