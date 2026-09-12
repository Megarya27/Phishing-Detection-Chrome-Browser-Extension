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
from sklearn.metrics import f1_score

#grid search is used to find best hyperparameters for XGBoost model instead of using default values. 
#hyperparameters tuned are max_depth and learning_rate. 
#goal is to find best combination of hyperparameters to maximize models performance
HYPERPARAMETER_GRID = [
    {"max_depth": max_depth, "learning_rate": learning_rate}
    for max_depth in (4, 6, 8)
    for learning_rate in (0.03, 0.05, 0.1)]

MAX_ESTIMATORS = 1000        #1000 maximum number of trees to build. Early stopping usually stops training before this
EARLY_STOPPING_ROUNDS = 30   #stop once validation logloss hasnt improved for this many rounds

#ensure that the minority class (phishing) is upweighted in training to account for class imbalance
def _class_balance_weight(y_train):
    y_train = np.asarray(y_train)
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    return float(n_neg) / float(n_pos) if n_pos > 0 else 1.0


def _fit_candidate(X_train, y_train, X_val, y_val, max_depth, learning_rate, seed):
    model = XGBClassifier(
        n_estimators=MAX_ESTIMATORS,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8, #fraction of training samples used for building each tree. It helps prevent overfitting by introducing randomness
                       #0.8 is common default to balance randomness and performance
        colsample_bytree=0.8,
        eval_metric="logloss", #logloss used as downstream code uses predicted probabilities for ROC-AUC and bootstrapping. 
                               #Logloss is standard metric for binary classification.
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        scale_pos_weight=_class_balance_weight(y_train),
        random_state=seed, #ensures reproducibility of results 
        n_jobs=-1)  #use all available CPU cores for faster training
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model

#searches over hyperparameter grid and returns best model, best hyperparameters, best validation F1 score and history of all grid points
def search_xgboost_hyperparameters(X_train, y_train, X_val, y_val, seed=RANDOM_SEED, grid=HYPERPARAMETER_GRID):
    best_model = None
    best_params = None
    best_f1 = -1.0
    history = {}
    for params in grid:
        model = _fit_candidate(X_train, y_train, X_val, y_val, seed=seed, **params)
        val_probs = model.predict_proba(X_val)[:, 1]
        val_preds = (val_probs >= 0.5).astype(int)
        val_f1 = f1_score(y_val, val_preds, zero_division=0)
        history[f"max_depth={params['max_depth']},learning_rate={params['learning_rate']}"] = {
            "val_f1": val_f1,
            "best_iteration": model.best_iteration}
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_params = {**params, "best_iteration": model.best_iteration}
            best_model = model

    return best_model, best_params, best_f1, history

#evaluates trained model on test set and returns predicted labels and probabilities
def evaluate(model, X, y): 
    y_prob = model.predict_proba(X)[:, 1]  #probabilities for class 1 (phishing)
    y_pred = (y_prob >= 0.5).astype(int)  #predicted labels based on probability threshold
                                          #threshold tuning done on hybrid model
    return y_pred, y_prob


def main():
    splits = load_structural_splits()  # defaults to all 30 UCI features

    model, best_params, val_f1, grid_history = search_xgboost_hyperparameters(
       splits["X_train"], splits["y_train"], splits["X_val"], splits["y_val"])
    print(f"Best hyperparameters: {best_params} (Val F1={val_f1:.4f})")

    y_pred, y_prob = evaluate(model, splits["X_test"], splits["y_test"])
    report, f1_dist = full_report("xgboost_structural", splits["y_test"], y_pred, y_prob)
    report["best_hyperparameters"] = best_params
    report["hyperparameter_grid_val_f1"] = grid_history
    print(json.dumps(report, indent=2))
    joblib.dump(model, MODELS_DIR / "baseline_xgboost.joblib")
    np.save(RESULTS_DIR / "baseline_xgboost_f1_bootstrap.npy", f1_dist) #raw bootstrap is saved for later analysis, e.g. confidence intervals, visualisation, etc.
    with open(RESULTS_DIR / "baseline_xgboost_report.json", "w") as f:
        json.dump(report, f, indent=2)

if __name__ == "__main__":
    main()
