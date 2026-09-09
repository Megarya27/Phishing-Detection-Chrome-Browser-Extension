# This program trains a baseline DistilBERT model on the phishing email dataset. 
# It uses Hugging Face Transformers library to fine-tune the pre-trained DistilBERT 
# model for binary classification (phishing vs legitimate emails). 
#The program also evaluates model's performance using metrics such as precision, 
# recall, F1 score, and ROC-AUC, and saves trained model and tokenizer for future use.

#Uses 30 UCI structural features for phishing website detection as to reproduce Shahrivari et al. (2020)
# and to provide a baseline for comparison with the hybrid model.
import json
import joblib
import numpy as np
from xgboost import XGBClassifier

from global_config import MODELS_DIR, RESULTS_DIR, RANDOM_SEED
from data_structural import load_structural_splits
from metrics import full_report


def train_xgboost(splits, seed=RANDOM_SEED):
    model = XGBClassifier(
        n_estimators=400, #number of trees in the ensemble, more trees can improve performance but increase training time
                          # too many trees can lead to overfitting, so 400 is a reasonable compromise

        max_depth=6,     #maximum depth of each tree, deeper trees can capture more complex patterns but may overfit
                         #6 is common default to balance complexity and generalisation

        learning_rate=0.05, #controls how much each tree contributes to the final prediction, smaller values can improve performance but require more trees
                            #0.05 is common default to balance learning speed and performance

        subsample=0.8,     #fraction of training samples used for building each tree. It helps prevent overfitting by introducing randomness
                           #0.8 is common default to balance randomness and performance
        colsample_bytree=0.8, 
        eval_metric="logloss", #logloss used as downstream code uses predicted probabilities for ROC-AUC and bootstrapping. 
                               #Logloss is standard metric for binary classification.
        random_state=seed, #ensures reproducibility of results 
        n_jobs=-1, #use all available CPU cores for faster training
    )
    #evaluate on validation set during training to monitor performance and prevent overfitting
    #early stopping is not used here, but could be added to stop training if validation performance stops improving.
    model.fit(
        splits["X_train"], splits["y_train"],
        eval_set=[(splits["X_val"], splits["y_val"])],
        verbose=False, #can be set to True for debugging 
    )
    return model

#evaluates trained model on test set and returns predicted labels and probabilities
def evaluate(model, X, y):
    y_prob = model.predict_proba(X)[:, 1] #probabilities for class 1 (phishing)
    y_pred = (y_prob >= 0.5).astype(int) #predicted labels based on probability threshold
                                         #threshold tuning done on hybrid model
    return y_pred, y_prob


def main():
    splits = load_structural_splits()  # defaults to all 30 UCI features
    model = train_xgboost(splits)

    y_pred, y_prob = evaluate(model, splits["X_test"], splits["y_test"])
    report, f1_dist = full_report("xgboost_structural", splits["y_test"], y_pred, y_prob)
    print(json.dumps(report, indent=2))

    joblib.dump(model, MODELS_DIR / "baseline_xgboost.joblib")
    np.save(RESULTS_DIR / "baseline_xgboost_f1_bootstrap.npy", f1_dist) #raw bootstrap is saved for later analysis, e.g. confidence intervals, visualisation, etc.
    with open(RESULTS_DIR / "baseline_xgboost_report.json", "w") as f:
        json.dump(report, f, indent=2) 


if __name__ == "__main__":
    main()
