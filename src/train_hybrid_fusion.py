# hybrid late-fusion ensemble combining distilBERT semantic predictions with XGBoost structural predictions
# run this after the base models have been trained and saved to disk
# uses dataset from build_email_url_paired_dataset.py, which pairs email text with URLs extracted from that email

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast, Trainer
from xgboost import XGBClassifier
from global_config import MODELS_DIR, PRE_CLICK_STRUCTURAL_FEATURES, RANDOM_SEED, RESULTS_DIR, ROOT
from metrics import bonferroni_correct, bootstrap_metric_distribution, confidence_interval, full_report, paired_bootstrap_ttest
from train_baseline_distilbert import predict_proba
from train_baseline_xgboost import search_xgboost_hyperparameters

#path to dataset where email text is paired with URL structural features
PAIRED_DATASET_PATH = ROOT / "prepared_datasets" / "paired_hybrid" / "paired_hybrid_email_urls.csv"

#minimum rows required to run hybrid evaluation
MIN_REAL_PAIRED_SAMPLES = 40
MIN_SAMPLES_FOR_INTERNAL_TUNING = 20 #min rows to carve a tuning fold for hyperparameter search
                                     #If fewer than this fallback to fixed hyperparameters
#blends probabilities linearly based on given weight
#weight is the proportion of semantic predictions. 1-weight is the proportion of structural predictions
#output between 0.0 and 1.0 inclusive as it is probability
def fuse_probabilities(semantic_probs, structural_probs, semantic_weight):
    return semantic_weight * semantic_probs + (1.0 - semantic_weight) * structural_probs

#grid searches both fusion blend weight and final decision threshold on validation split
def tune_fusion_weight_and_threshold(val_labels, val_semantic_probs, val_structural_probs, weight_step=0.05, threshold_step=0.05):
    # tracks best combination of weight and classification cutoff
    best = {"weight": 0.5, "threshold": 0.5, "f1": -1.0}
    history = {}

    # step weights from 0.0 to 1.0 inclusive
    #small epsilon (1e-9) ensures endpoint 1.0 is included despite floating point drift
    for weight in np.arange(0.0, 1.0 + 1e-9, weight_step):
        # clean floating point inaccuracies like 0.300000000004 into 0.3.
        weight = round(float(weight), 2)
        fused = fuse_probabilities(val_semantic_probs, val_structural_probs, weight)
        # restrict threshold to (0.05, 0.95) inclusive, to avoid all positive/all negative splits
        for threshold in np.arange(0.05, 0.95 + 1e-9, threshold_step):
            clean_threshold = round(float(threshold), 2)
            #convert probability into a binary label based on current threshold
            preds = (fused >= clean_threshold).astype(int)
            # zero_division=0 prevents error and sets score to 0 if no positive samples are predicted.
            f1 = f1_score(val_labels, preds, zero_division=0)
            #store result with tuple key for reference
            history[(weight, clean_threshold)] = f1
            # save highest score found so far.
            if f1 > best["f1"]:
                best = {"weight": weight, "threshold": clean_threshold, "f1": f1}
    return best["weight"], best["threshold"], best["f1"], history


# trains a logistic regression meta-classifier over [structural, semantic] on validation data
#automatically learns how much to trust each model instead of guessing fixed weight
def train_meta_fusion_classifier(val_labels, val_semantic_probs, val_structural_probs):
    #stacks probabilities horizontally to form 2 column feature matrix [structural, semantic]
    #keep column ordering consistent 
    features = np.column_stack([val_structural_probs, val_semantic_probs])
    model = LogisticRegression()
    model.fit(features, val_labels)
    return model


# finds decision threshold on validation data to maximise F1 score
def tune_threshold(val_labels, predicted_probs, step_size=0.05):
    #initialise with neutral 50/50 weight and impossible low f1 score
    best_threshold, best_f1 = 0.5, -1.0
    # 0.05 to 0.95 inclusive to avoid all positive/all negative splits
    for threshold in np.arange(0.05, 0.95 + 1e-9, step_size):
        # clean floating point inaccuracies like 0.300000000004 into 0.3.
        threshold = round(float(threshold), 2)
        # convert continuous probabilities to binary 0 or 1 using current threshold
        preds = (predicted_probs >= threshold).astype(int)
        # zero_division=0 prevents error and sets score to 0 if no positive samples are predicted
        f1 = f1_score(val_labels, preds, zero_division=0)
        # save highest score found so far
        if f1 > best_f1:
            best_f1, best_threshold = f1, threshold
    return best_threshold, best_f1


# loads paired dataset and drops incomplete rows instead of filling with fake values.
def load_real_paired_dataset(dataset_path: Path = PAIRED_DATASET_PATH):
    # return None early if file is not on disk yet.
    if not dataset_path.exists():
        return None

    df = pd.read_csv(dataset_path)
    #only keep rows where the crawl status is "ok" to avoid dead links
    if "status" in df.columns:
        df = df[df["status"] == "ok"]

    #the paired hybrid model uses only preclick features. destination-DOM
    # features are not used in this real time data path.
    df = df.dropna(subset=["label"] + PRE_CLICK_STRUCTURAL_FEATURES + ["text"])
    #verify we have enough data and both classes (0 and 1) are present
    if len(df) < MIN_REAL_PAIRED_SAMPLES or df["label"].nunique() < 2:
        return None
    #reset indices cleanly after dropping rows
    return df.reset_index(drop=True)

#trains XGBoost model directly on successfully crawled email-URL pairs for domain-matched structural predictions
def train_domain_matched_structural_model(train_features, train_labels, random_seed=RANDOM_SEED):
    if len(train_features) < MIN_SAMPLES_FOR_INTERNAL_TUNING:
        print(
            f"Only {len(train_features)} paired training rows available - too few to carve "
            f"out an internal tuning fold, falling back to fixed hyperparameters."
        )
        # use fixed fallback configuration when the paired training set is too small for internal tuning
        model = XGBClassifier(
            n_estimators=400, #400 trees
            max_depth=6, 
            learning_rate=0.05,
            subsample=0.8, #0.8 of rows per tree
            colsample_bytree=0.8, 
            eval_metric="logloss", #evaluate on log loss during training
            random_state=random_seed, #ensureds reproducible results
            n_jobs=-1, #uses all CPU cores for training
        )
        # fit only on training split to prevent data leakage into validation/test split
        #no evaluation set is used as the validation split is used for threshold tuning and metaclassifier training.
        model.fit(train_features, train_labels)
        return model

    fit_features, tune_features, fit_labels, tune_labels = train_test_split(
        train_features, train_labels, test_size=0.15, stratify=train_labels, random_state=random_seed
    )
    model, best_params, tune_f1, _ = search_xgboost_hyperparameters(
        fit_features, fit_labels, tune_features, tune_labels, seed=random_seed
    )
    print(f"Domain-matched structural model: best hyperparameters {best_params} (tuning-fold F1={tune_f1:.4f})")
    return model


# loads tokenizer and model from disk into trainer object for batched predictions
def load_distilbert_trainer(model_directory: Path):
    #tokenizer and model must load from same directory to guarantee matching vocab IDs
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_directory)
    model = DistilBertForSequenceClassification.from_pretrained(model_directory)
    # bare trainer without training args as it is only used for inference
    trainer = Trainer(model=model)
    return trainer, tokenizer


#function to load saved distilBERT model:
def load_models():
    return load_distilbert_trainer(MODELS_DIR / "distilbert_semantic" / "final")


# splits paired data 70/15/15 into train, validation, and test
# trains the structural model on train and generates predictions for val and test splits
def split_real_paired_data(paired_df, trainer, tokenizer):
    #use only features available before clicking the link (no live DOM features)
    structural_features = paired_df[PRE_CLICK_STRUCTURAL_FEATURES].astype(float)
    labels = paired_df["label"].astype(int).to_numpy()
    #run batch inference with distilBERT over all email text rows
    semantic_probs = predict_proba(trainer, tokenizer, paired_df["text"].tolist())

    #stratified split: carves out first 30% for validation + test combined
    train_idx, remaining_idx = train_test_split(
        np.arange(len(paired_df)), test_size=0.30, stratify=labels, random_state=RANDOM_SEED
    )
    #split remaining 30% evenly (50/50) into 15% validation and 15% test
    val_idx, test_idx = train_test_split(
        remaining_idx, test_size=0.50, stratify=labels[remaining_idx], random_state=RANDOM_SEED)

    #fit domain matched structural model on training indices only
    structural_model = train_domain_matched_structural_model(
        structural_features.iloc[train_idx], labels[train_idx])
    #save trained model artifact to disk
    joblib.dump(structural_model, MODELS_DIR / "structural_pre_click_paired.joblib")

    #extract structural prediction probabilities for the positive phishing class (column 1)
    structural_probs = structural_model.predict_proba(structural_features)[:, 1]

    #return sliced arrays partitioned by split indices
    return {
        "evaluated_on": "real_paired_crawl_data",
        "p_text_val": semantic_probs[val_idx],
        "p_struct_val": structural_probs[val_idx],
        "y_val": labels[val_idx],
        "p_text_test": semantic_probs[test_idx],
        "p_struct_test": structural_probs[test_idx],
        "y_test": labels[test_idx]}


#loads dataset and raises error if missing/too small
def get_paired_splits(trainer, tokenizer):
    paired_df = load_real_paired_dataset()
    #stops execution if paired dataset doesnt exist
    if paired_df is None:
        raise RuntimeError(
            f"No usable paired dataset found at {PAIRED_DATASET_PATH} "
            f"(need >= {MIN_REAL_PAIRED_SAMPLES} rows with status='ok' and both classes) - "
            f"run build_email_url_paired_dataset.py first")

    print(f"Loaded {len(paired_df)} paired samples from {PAIRED_DATASET_PATH}.")
    return split_real_paired_data(paired_df, trainer, tokenizer)


#runs reference grid search to log the baseline methodology for comparison
def run_reference_grid_search(data_splits):
    weight, threshold, val_f1, history = tune_fusion_weight_and_threshold(
        data_splits["y_val"], data_splits["p_text_val"], data_splits["p_struct_val"]
    )
    print(f"[Reference] Grid search picked weight={weight}, threshold={threshold} (Val F1={val_f1:.4f})")
    return weight, threshold, val_f1, history


#trains meta classifier on validation, tunes the threshold on validation, and evaluates on test.
def fit_and_evaluate_meta_fusion(data_splits):
    # unpack validation split arrays.
    val_labels = data_splits["y_val"]
    val_semantic = data_splits["p_text_val"]
    val_structural = data_splits["p_struct_val"]

    #unpack test split arrays
    test_labels = data_splits["y_test"]
    test_semantic = data_splits["p_text_test"]
    test_structural = data_splits["p_struct_test"]

    #train meta classifier and optimise threshold on validation split
    meta_classifier = train_meta_fusion_classifier(val_labels, val_semantic, val_structural)

    #get fused probabilities for validation to find best operating threshold
    val_features = np.column_stack([val_structural, val_semantic])
    val_fused = meta_classifier.predict_proba(val_features)[:, 1]
    threshold, val_f1 = tune_threshold(val_labels, val_fused)

    #print learned coefficients: coef_[0][0] is structural, coef_[0][1] is semantic.
    print(
        f"Meta fusion weights: structural={meta_classifier.coef_[0][0]:.4f}, "
        f"semantic={meta_classifier.coef_[0][1]:.4f}, intercept={meta_classifier.intercept_[0]:.4f} | "
        f"Threshold={threshold} (Val F1={val_f1:.4f})"
    )

    #evaluate on heldout test split using threshold learned from validation
    test_features = np.column_stack([test_structural, test_semantic])
    test_fused = meta_classifier.predict_proba(test_features)[:, 1]
    test_preds = (test_fused >= threshold).astype(int)

    #generate metrics dictionary and bootstrap distribution for test set
    report, f1_distribution = full_report("hybrid_meta_fusion", test_labels, test_preds, test_fused)
    #add experiment metadata to final report
    report.update({
        "fusion_method": "logistic_regression_meta_classifier",
        "meta_classifier_coefficients": {
            "structural_weight": float(meta_classifier.coef_[0][0]),
            "semantic_weight": float(meta_classifier.coef_[0][1]),
            "intercept": float(meta_classifier.intercept_[0]),
        },
        "decision_threshold": threshold,
        "threshold_objective": "Maximized F1 on validation split; never tuned on test set",
        "evaluated_on": "real_paired_crawl_data",
        "structural_model_used": "structural_pre_click_paired.joblib (trained on pre-click features)"})

    print(json.dumps(report, indent=2))
    #save meta classifier to disk for future inference without retraining
    joblib.dump(meta_classifier, MODELS_DIR / "hybrid_meta_fusion.joblib")

    return {
        "meta_clf": meta_classifier,
        "hybrid_report": report,
        "hybrid_f1_dist": f1_distribution,
        "p_fused_test": test_fused,
        "y_pred_test": test_preds}

#evaluates structural only and semantic only baselines on same test split using their own tuned thresholds
def evaluate_single_modality_baselines(data_splits):
    val_labels = data_splits["y_val"]
    val_structural = data_splits["p_struct_val"]
    val_semantic = data_splits["p_text_val"]

    test_labels = data_splits["y_test"]
    test_structural = data_splits["p_struct_test"]
    test_semantic = data_splits["p_text_test"]

    #structural baseline (XGBoost): tunes its cutoff on validation, then apply to test
    structural_threshold, _ = tune_threshold(val_labels, val_structural)
    structural_preds = (test_structural >= structural_threshold).astype(int)
    structural_report, structural_f1_dist = full_report(
        "xgboost_structural_only_on_paired_test", test_labels, structural_preds, test_structural)
    structural_report["decision_threshold"] = structural_threshold

    #semantic baseline (distilBERT): tunes its cutoff on validation, then apply to test
    semantic_threshold, _ = tune_threshold(val_labels, val_semantic)
    semantic_preds = (test_semantic >= semantic_threshold).astype(int)
    semantic_report, semantic_f1_dist = full_report(
        "distilbert_semantic_only_on_paired_test", test_labels, semantic_preds, test_semantic)
    semantic_report["decision_threshold"] = semantic_threshold

    print("\nSingle Modality Baselines (Same Test Split):")
    print(json.dumps(structural_report, indent=2))
    print(json.dumps(semantic_report, indent=2))

    return {
        "structural_only_report": structural_report,
        "structural_only_f1_dist": structural_f1_dist,
        "y_pred_struct_only": structural_preds,
        "semantic_only_report": semantic_report,
        "semantic_only_f1_dist": semantic_f1_dist,
        "y_pred_text_only": semantic_preds}


#runs paired bootstrap t tests to see if hybrid is significantly better than the best baseline
def run_significance_tests(data_splits, hybrid_results, baseline_results):
    test_labels = data_splits["y_test"]
    test_structural = data_splits["p_struct_test"]
    test_semantic = data_splits["p_text_test"]

    #finds stronger baseline by checking which has higher mean F1 score in bootstrap distribution
    structural_f1_mean = np.nanmean(baseline_results["structural_only_f1_dist"])
    semantic_f1_mean = np.nanmean(baseline_results["semantic_only_f1_dist"])
    if structural_f1_mean >= semantic_f1_mean:
        best_baseline_name = "xgboost_structural_only"
        best_baseline_f1_dist = baseline_results["structural_only_f1_dist"]
    else:
        best_baseline_name = "distilbert_semantic_only"
        best_baseline_f1_dist = baseline_results["semantic_only_f1_dist"]

    #paired t test between hybrid and best baseline bootstrap distributions.
    t_stat_f1, p_value_f1 = paired_bootstrap_ttest(hybrid_results["hybrid_f1_dist"], best_baseline_f1_dist)
    print(f"\nSignificance vs {best_baseline_name} (F1): t={t_stat_f1:.4f}, p={p_value_f1:.6f}")

    #compute bootstrap distributions for AUC-ROC for hybrid and both baselines.
    hybrid_auc_dist = bootstrap_metric_distribution(
        test_labels, hybrid_results["y_pred_test"], hybrid_results["p_fused_test"], metric="auc_roc")
    structural_auc_dist = bootstrap_metric_distribution(
        test_labels, baseline_results["y_pred_struct_only"], test_structural, metric="auc_roc")
    semantic_auc_dist = bootstrap_metric_distribution(
        test_labels, baseline_results["y_pred_text_only"], test_semantic, metric="auc_roc")

    #find the best baseline on AUC-ROC ranking metric
    structural_auc_mean = np.nanmean(structural_auc_dist)
    semantic_auc_mean = np.nanmean(semantic_auc_dist)
    if structural_auc_mean >= semantic_auc_mean:
        best_baseline_name_auc = "xgboost_structural_only"
        best_baseline_auc_dist = structural_auc_dist
    else:
        best_baseline_name_auc = "distilbert_semantic_only"
        best_baseline_auc_dist = semantic_auc_dist
    #run paired t-test on AUC-ROC distributions
    t_stat_auc, p_value_auc = paired_bootstrap_ttest(hybrid_auc_dist, best_baseline_auc_dist)
    # calculate 95% confidence interval for hybrid AUC-ROC.
    auc_ci_low, auc_ci_high = confidence_interval(hybrid_auc_dist)
    print(f"Significance vs {best_baseline_name_auc} (AUC-ROC): t={t_stat_auc:.4f}, p={p_value_auc:.6f}")

    # bonferroni correction for testing two metrics (F1 and AUC-ROC) at alpha=0.05.
    corrected_alpha, (f1_significant, auc_significant) = bonferroni_correct([p_value_f1, p_value_auc], alpha=0.05)
    print(
        f"Bonferroni-corrected alpha: {corrected_alpha:.4f} | "
        f"F1 Significant: {f1_significant} | AUC-ROC Significant: {auc_significant}")

    return {
        "best_baseline_name": best_baseline_name,
        "t_stat": t_stat_f1,
        "p_value": p_value_f1,
        "best_baseline_name_auc": best_baseline_name_auc,
        "t_stat_auc": t_stat_auc,
        "p_value_auc": p_value_auc,
        "auc_ci_low": auc_ci_low,
        "auc_ci_high": auc_ci_high,
        "corrected_alpha": corrected_alpha,
        "f1_significant": f1_significant,
        "auc_significant": auc_significant}


#compiles metrics, grid search logs, and significance test results into one dictionary.
def assemble_results(grid_search_results, hybrid_results, baseline_results, significance_results):
    grid_weight, grid_threshold, grid_val_f1, weight_score_history = grid_search_results

    return {
        "hybrid_report": hybrid_results["hybrid_report"],
        "manual_grid_search_reference_only": {
            "note": "Reference only: manual linear blend grid-search. Not used for primary hybrid 1.",
            "weight": grid_weight,
            "threshold": grid_threshold,
            "val_f1": grid_val_f1,
        },
        # dictionary keys must be string format for valid JSON serializing.
        "fusion_weight_threshold_grid_scores": {
            f"weight={weight},threshold={threshold}": score
            for (weight, threshold), score in weight_score_history.items()
        },
        "single_modality_baselines_on_same_test_population": {
            "structural_only": baseline_results["structural_only_report"],
            "semantic_only": baseline_results["semantic_only_report"],
        },
        "significance_test_f1": {
            "compared_against": significance_results["best_baseline_name"],
            "t_stat": significance_results["t_stat"],
            "p_value": significance_results["p_value"],
            "bonferroni_corrected_alpha": significance_results["corrected_alpha"],
            "significant_after_correction": significance_results["f1_significant"],
        },
        "significance_test_auc_roc": {
            "compared_against": significance_results["best_baseline_name_auc"],
            "hybrid_auc_ci_low": significance_results["auc_ci_low"],
            "hybrid_auc_ci_high": significance_results["auc_ci_high"],
            "t_stat": significance_results["t_stat_auc"],
            "p_value": significance_results["p_value_auc"],
            "bonferroni_corrected_alpha": significance_results["corrected_alpha"],
            "significant_after_correction": significance_results["auc_significant"],
        },
    }


#saves final report to JSON and saves bootstrap distribution array
def save_results(results, hybrid_f1_distribution):
    # write formatted JSON report to disk.
    with open(RESULTS_DIR / "hybrid_fusion_report.json", "w") as f:
        json.dump(results, f, indent=2)
    #save the hybrid F1 bootstrap distribution for later confidence-interval analysis
    np.save(RESULTS_DIR / "hybrid_f1_bootstrap.npy", hybrid_f1_distribution)


def main():
    # load pretrained distilBERT components
    trainer, tokenizer = load_models()
    # partition paired dataset and extract predictions
    data_splits = get_paired_splits(trainer, tokenizer)
    #compute reference grid search.
    grid_search_results = run_reference_grid_search(data_splits)
    #fit meta fusion classifier and evaluate on testset
    hybrid_results = fit_and_evaluate_meta_fusion(data_splits)
    #evaluate single modality baseline on same test set
    baseline_results = evaluate_single_modality_baselines(data_splits)
    #run bootstrap significance tests comparing hybrid to best baseline
    significance_results = run_significance_tests(data_splits, hybrid_results, baseline_results)

    #package and save all reports to disk.
    final_results = assemble_results(grid_search_results, hybrid_results, baseline_results, significance_results)
    save_results(final_results, hybrid_results["hybrid_f1_dist"])


if __name__ == "__main__":
    main()