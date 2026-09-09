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

from global_config import (MODELS_DIR,PRE_CLICK_STRUCTURAL_FEATURES,RANDOM_SEED,RESULTS_DIR,ROOT)
from metrics import ( bonferroni_correct,bootstrap_metric_distribution,confidence_interval,full_report,paired_bootstrap_ttest)
from train_baseline_distilbert import predict_proba

#path to dataset where email text is paired with URL structural features
PAIRED_DATASET_PATH = ROOT / "prepared_datasets" / "paired_hybrid" / "paired_hybrid_email_urls.csv"

#minimum rows required to train hybrid model
MIN_REAL_PAIRED_SAMPLES = 40

#blends probabilities linearly based on given weight
#weight is the proportion of semantic predictions. 1 weight is the proportion of structural predictions.
#output between 0.0 and 1.0 inclusive as it is probability
def fuse_probabilities(semantic_probs: np.ndarray, structural_probs: np.ndarray, semantic_weight: float) -> np.ndarray:
    return semantic_weight * semantic_probs + (1.0 - semantic_weight) * structural_probs


#grid searches for best mixing weight on validation data using fixed 0.5 decision threshold.
#0.0 is structural only, 0.5 is equal weight, 1.0 is semantic only.
def tune_fusion_weight(
    val_labels: np.ndarray,
    val_semantic_probs: np.ndarray,
    val_structural_probs: np.ndarray,
    step_size: float = 0.05,
):
    #initialise with neutral 50/50 weight and impossible low f1 score.
    best_weight = 0.5
    best_f1_score = -1.0
    weight_score_history = {}

    #small epsilon (1e-9) ensures endpoint 1.0 is included despite floating point drift.
    for weight in np.arange(0.0, 1.0 + 1e-9, step_size):
        # clean floating point inaccuracies like 0.300000000004 into 0.3.
        clean_weight = round(float(weight), 2)
        fused_probs = fuse_probabilities(val_semantic_probs, val_structural_probs, clean_weight)
        #0.5 threshold is used for all weights to isolate the effect of the weight itself.
        predicted_labels = (fused_probs >= 0.5).astype(int)
        # zero_division=0 prevents error and sets score to 0 if no positive samples are predicted.
        current_f1 = f1_score(val_labels, predicted_labels, zero_division=0)
        weight_score_history[clean_weight] = current_f1

        # save highest score found so far.
        if current_f1 > best_f1_score:
            best_f1_score = current_f1
            best_weight = clean_weight

    return best_weight, best_f1_score, weight_score_history


#grid searches both fusion blend weight and final decision threshold on validation split
def tune_fusion_weight_and_threshold(val_labels: np.ndarray,val_semantic_probs: np.ndarray,val_structural_probs: np.ndarray,
    weight_step: float = 0.05,threshold_step: float = 0.05):
    # tracks best combination of weight and classification cutoff
    best_config = {"weight": 0.5, "threshold": 0.5, "f1": -1.0}
    grid_score_history = {}
    # step weights from 0.0 to 1.0 inclusive
    for weight in np.arange(0.0, 1.0 + 1e-9, weight_step):
        clean_weight = round(float(weight), 2)
        fused_probs = fuse_probabilities(val_semantic_probs, val_structural_probs, clean_weight)

        # restrict threshold to (0.05, 0.95) inclusive, to avoid all positive/all negative splits
        for threshold in np.arange(0.05, 0.95 + 1e-9, threshold_step):
            clean_threshold = round(float(threshold), 2)
            #convert probability into a binary label based on current threshold
            predicted_labels = (fused_probs >= clean_threshold).astype(int)
            current_f1 = f1_score(val_labels, predicted_labels, zero_division=0)
            #store result with tuple key for reference
            grid_score_history[(clean_weight, clean_threshold)] = current_f1

            if current_f1 > best_config["f1"]:
                best_config = {"weight": clean_weight, "threshold": clean_threshold, "f1": current_f1}

    return best_config["weight"], best_config["threshold"], best_config["f1"], grid_score_history


# trains a logistic regression meta-classifier over [structural, semantic] on validation data
#automatically learns how much to trust each model instead of guessing fixed weight
def train_meta_fusion_classifier(val_labels: np.ndarray,val_semantic_probs: np.ndarray,val_structural_probs: np.ndarray):
    #stacks probabilities horizontally to form 2 column feature matrix [structural, semantic]
    #keep column ordering consistent 
    validation_features = np.column_stack([val_structural_probs, val_semantic_probs])

    #fit on validation probabilities so it learns generalisation error, not training fits.
    meta_model = LogisticRegression()
    meta_model.fit(validation_features, val_labels)
    return meta_model


# finds decision threshold on validation data to maximise F1 score
def tune_threshold(val_labels: np.ndarray, predicted_probs: np.ndarray, step_size: float = 0.05):
    best_threshold = 0.5
    best_f1_score = -1.0

    # 0.05 to 0.95 inclusive to avoid all positive/all negative splits
    for threshold in np.arange(0.05, 0.95 + 1e-9, step_size):
        clean_threshold = round(float(threshold), 2)
        # convert continuous probabilities to binary 0 or 1 using current threshold
        current_predictions = (predicted_probs >= clean_threshold).astype(int)
        current_f1 = f1_score(val_labels, current_predictions, zero_division=0)
        if current_f1 > best_f1_score:
            best_f1_score = current_f1
            best_threshold = clean_threshold

    return best_threshold, best_f1_score


# loads paired dataset and drops incomplete rows instead of filling with fake values.
def load_real_paired_dataset(dataset_path: Path = PAIRED_DATASET_PATH) -> pd.DataFrame | None:
    # return None early if file is not on disk yet.
    if not dataset_path.exists():
        return None

    paired_dataframe = pd.read_csv(dataset_path)
    #only keep rows where the crawl status is "ok" to avoid dead links
    if "status" in paired_dataframe.columns:
        paired_dataframe = paired_dataframe[paired_dataframe["status"] == "ok"]

    #the paired hybrid model uses only preclick features. destination-DOM
    # features are not used in this real time data path.
    paired_dataframe = paired_dataframe.dropna(subset=["label"] + PRE_CLICK_STRUCTURAL_FEATURES + ["text"])

    #verify we have enough data and both classes (0 and 1) are present
    if len(paired_dataframe) < MIN_REAL_PAIRED_SAMPLES or paired_dataframe["label"].nunique() < 2:
        return None

    #reset indices cleanly after dropping rows
    return paired_dataframe.reset_index(drop=True)


#trains XGBoost model directly on email URL data so it learns how to handle dead links.
def train_domain_matched_structural_model(train_features: pd.DataFrame,train_labels: np.ndarray,
     random_seed: int = RANDOM_SEED):
    # matching the baseline hyperparameters exactly so differences come from data, not tuning.
    structural_model = XGBClassifier(
        n_estimators=400, #400 trees
        max_depth=6,
        learning_rate=0.05, 
        subsample=0.8, #0.8 of rows per tree
        colsample_bytree=0.8, 
        eval_metric="logloss", #evaluate on log loss during training
        random_state=random_seed, #ensureds reproducible results
        n_jobs=-1) #uses all CPU cores for training
    # fit only on training split to prevent data leakage into validation/test split
    #no evaluation set is used as the validation split is used for threshold tuning and metaclassifier training.
    structural_model.fit(train_features, train_labels)
    return structural_model


# loads tokenizer and model from disk into trainer object for batched predictions
def load_distilbert_trainer(model_directory: Path):
    #tokenizer and model must load from same directory to guarantee matching vocab IDs
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_directory)
    model = DistilBertForSequenceClassification.from_pretrained(model_directory)
    # bare trainer without training args as it is only used for inference
    evaluation_trainer = Trainer(model=model)
    return evaluation_trainer, tokenizer


#function to load saved distilBERT model:
def load_models():
    semantic_model_directory = MODELS_DIR / "distilbert_semantic" / "final"
    return load_distilbert_trainer(semantic_model_directory)


# splits paired data 70/15/15 into train, validation, and test
# trains the structural model on train and generates predictions for val and test splits
def split_real_paired_data(paired_dataframe: pd.DataFrame,trainer: Trainer,tokenizer: DistilBertTokenizerFast):
    #use only features available before clicking the link (no live DOM features)
    structural_features = paired_dataframe[PRE_CLICK_STRUCTURAL_FEATURES].astype(float)
    ground_truth_labels = paired_dataframe["label"].astype(int).to_numpy()
    #run batch inference with distilBERT over all email text rows
    all_semantic_probs = predict_proba(trainer, tokenizer, paired_dataframe["text"].tolist())

    #stratified split: carves out first 30% for validation + test combined
    train_indices, remaining_indices = train_test_split(
        np.arange(len(paired_dataframe)),
        test_size=0.30,
        stratify=ground_truth_labels,
        random_state=RANDOM_SEED)
    #split remaining 30% evenly (50/50) into 15% validation and 15% test
    val_indices, test_indices = train_test_split(
        remaining_indices,
        test_size=0.50,
        stratify=ground_truth_labels[remaining_indices],
        random_state=RANDOM_SEED)

    #fit domain matched structural model on training indices only
    domain_structural_model = train_domain_matched_structural_model(
        structural_features.iloc[train_indices],
        ground_truth_labels[train_indices])
    
    #save trained model artifact to disk
    joblib.dump(domain_structural_model, MODELS_DIR / "structural_pre_click_paired.joblib")

    #extract structural prediction probabilities for the positive phishing class (column 1)
    all_structural_probs = domain_structural_model.predict_proba(structural_features)[:, 1]

    #return sliced arrays partitioned by split indices
    return {
        "evaluated_on": "real_paired_crawl_data",
        "p_text_val": all_semantic_probs[val_indices],
        "p_struct_val": all_structural_probs[val_indices],
        "y_val": ground_truth_labels[val_indices],
        "p_text_test": all_semantic_probs[test_indices],
        "p_struct_test": all_structural_probs[test_indices],
        "y_test": ground_truth_labels[test_indices] }


#loads dataset and raises error if missing/too small
def get_paired_splits(trainer: Trainer, tokenizer: DistilBertTokenizerFast):
    paired_dataframe = load_real_paired_dataset()
    #stops execution if paired dataset doesnt exist
    if paired_dataframe is None:
        raise RuntimeError(
            f"No usable paired dataset found at {PAIRED_DATASET_PATH} "
            f"(need >= {MIN_REAL_PAIRED_SAMPLES} rows with status='ok' and both classes)"
            f"run build_email_url_paired_dataset.py first"
        )

    print(f"Loaded {len(paired_dataframe)} paired samples from {PAIRED_DATASET_PATH}.")
    return split_real_paired_data(paired_dataframe, trainer, tokenizer)

#runs reference grid search to log the baseline methodology for comparison
def run_reference_grid_search(data_splits: dict):
    best_weight, best_threshold, best_val_f1, score_history = tune_fusion_weight_and_threshold(
        data_splits["y_val"],
        data_splits["p_text_val"],
        data_splits["p_struct_val"])
    print(f"[Reference] Grid search picked weight={best_weight}, threshold={best_threshold} (Val F1={best_val_f1:.4f})")
    return best_weight, best_threshold, best_val_f1, score_history

#trains meta classifier on validation, tunes the threshold on validation, and evaluates on test.
def fit_and_evaluate_meta_fusion(data_splits: dict):
    # unpack validation split arrays.
    val_labels = data_splits["y_val"]
    val_semantic_probs = data_splits["p_text_val"]
    val_structural_probs = data_splits["p_struct_val"]

    #unpack test split arrays
    test_labels = data_splits["y_test"]
    test_semantic_probs = data_splits["p_text_test"]
    test_structural_probs = data_splits["p_struct_test"]

    #train meta classifier and optimise threshold on validation split
    meta_classifier = train_meta_fusion_classifier(val_labels, val_semantic_probs, val_structural_probs)

    #get fused probabilities for validation to find best operating threshold
    val_features = np.column_stack([val_structural_probs, val_semantic_probs])
    val_fused_probs = meta_classifier.predict_proba(val_features)[:, 1]
    best_threshold, best_val_f1 = tune_threshold(val_labels, val_fused_probs)

    #print learned coefficients: coef_[0][0] is structural, coef_[0][1] is semantic.
    print(f"Meta fusion weights: structural={meta_classifier.coef_[0][0]:.4f}, "
        f"semantic={meta_classifier.coef_[0][1]:.4f}, intercept={meta_classifier.intercept_[0]:.4f} | "
        f"Threshold={best_threshold} (Val F1={best_val_f1:.4f})"
    )

    #evaluate on heldout test split using threshold learned from validation
    test_features = np.column_stack([test_structural_probs, test_semantic_probs])
    test_fused_probs = meta_classifier.predict_proba(test_features)[:, 1]
    test_predictions = (test_fused_probs >= best_threshold).astype(int)

    #generate metrics dictionary and bootstrap distribution for test set
    hybrid_report, hybrid_f1_distribution = full_report(
        "hybrid_meta_fusion", test_labels, test_predictions, test_fused_probs)
    #add experiment metadata to final report
    hybrid_report.update({
        "fusion_method": "logistic_regression_meta_classifier",
        "meta_classifier_coefficients": {
            "structural_weight": float(meta_classifier.coef_[0][0]),
            "semantic_weight": float(meta_classifier.coef_[0][1]),
            "intercept": float(meta_classifier.intercept_[0])},

        "decision_threshold": best_threshold,
        "threshold_objective": "Maximized F1 on validation split; never tuned on test set",
        "evaluated_on": "real_paired_crawl_data",
        "structural_model_used": "structural_pre_click_paired.joblib (trained on pre-click features)"})
    
    print(json.dumps(hybrid_report, indent=2))
    #save meta classifier to disk for future inference without retraining
    joblib.dump(meta_classifier, MODELS_DIR / "hybrid_meta_fusion.joblib")

    return {
        "meta_clf": meta_classifier,
        "hybrid_report": hybrid_report,
        "hybrid_f1_dist": hybrid_f1_distribution,
        "p_fused_test": test_fused_probs,
        "y_pred_test": test_predictions }


#evaluates structural only and semantic only baselines on same test split using their own tuned thresholds
def evaluate_single_modality_baselines(data_splits: dict):
    val_labels = data_splits["y_val"]
    val_structural_probs = data_splits["p_struct_val"]
    val_semantic_probs = data_splits["p_text_val"]

    test_labels = data_splits["y_test"]
    test_structural_probs = data_splits["p_struct_test"]
    test_semantic_probs = data_splits["p_text_test"]

    #structural baseline (XGBoost): tunes its cutoff on validation, then apply to test
    structural_threshold, _ = tune_threshold(val_labels, val_structural_probs)
    test_structural_predictions = (test_structural_probs >= structural_threshold).astype(int)
    structural_report, structural_f1_distribution = full_report(
        "xgboost_structural_only_on_paired_test",
        test_labels,
        test_structural_predictions,
        test_structural_probs)
    structural_report["decision_threshold"] = structural_threshold

    #semantic baseline (distilBERT): tunes its cutoff on validation, then apply to test
    semantic_threshold, _ = tune_threshold(val_labels, val_semantic_probs)
    test_semantic_predictions = (test_semantic_probs >= semantic_threshold).astype(int)
    semantic_report, semantic_f1_distribution = full_report("distilbert_semantic_only_on_paired_test",
        test_labels,
        test_semantic_predictions,
        test_semantic_probs )
    semantic_report["decision_threshold"] = semantic_threshold

    print("\n Single Modality Baselines (Same Test Split):")
    print(json.dumps(structural_report, indent=2))
    print(json.dumps(semantic_report, indent=2))

    return {
        "structural_only_report": structural_report,
        "structural_only_f1_dist": structural_f1_distribution,
        "y_pred_struct_only": test_structural_predictions,
        "semantic_only_report": semantic_report,
        "semantic_only_f1_dist": semantic_f1_distribution,
        "y_pred_text_only": test_semantic_predictions }


#runs paired bootstrap t tests to see if hybrid is significantly better than the best baseline
def run_significance_tests(data_splits: dict, hybrid_results: dict, baseline_results: dict):
    test_labels = data_splits["y_test"]
    test_structural_probs = data_splits["p_struct_test"]
    test_semantic_probs = data_splits["p_text_test"]

    #finds stronger baseline by checking which has higher mean F1 score in bootstrap distribution
    best_baseline_name, best_baseline_f1_distribution = max(
        [("xgboost_structural_only", baseline_results["structural_only_f1_dist"]),
        ("distilbert_semantic_only", baseline_results["semantic_only_f1_dist"])],
        key=lambda item: np.nanmean(item[1]))
    
    #paired t test between hybrid and best baseline bootstrap distributions.
    t_stat_f1, p_value_f1 = paired_bootstrap_ttest(
        hybrid_results["hybrid_f1_dist"],
        best_baseline_f1_distribution,
    )
    print(f"\nSignificance vs {best_baseline_name} (F1): t={t_stat_f1:.4f}, p={p_value_f1:.6f}")

    #compute bootstrap distributions for AUC-ROC for hybrid and both baselines.
    hybrid_auc_distribution = bootstrap_metric_distribution(
        test_labels,
        hybrid_results["y_pred_test"],
        hybrid_results["p_fused_test"],
        metric="auc_roc",
    )
    structural_auc_distribution = bootstrap_metric_distribution(
        test_labels,
        baseline_results["y_pred_struct_only"],
        test_structural_probs,
        metric="auc_roc",
    )
    semantic_auc_distribution = bootstrap_metric_distribution(
        test_labels,
        baseline_results["y_pred_text_only"],
        test_semantic_probs,
        metric="auc_roc",
    )

    #find the best baseline on AUC-ROC ranking metric
    best_baseline_name_auc, best_baseline_auc_distribution = max(
        [
            ("xgboost_structural_only", structural_auc_distribution),
            ("distilbert_semantic_only", semantic_auc_distribution),
        ],
        key=lambda item: np.nanmean(item[1]),
    )
    #run paired t-test on AUC-ROC distributions
    t_stat_auc, p_value_auc = paired_bootstrap_ttest(
        hybrid_auc_distribution,
        best_baseline_auc_distribution,
    )
    # calculate 95% confidence interval for hybrid AUC-ROC.
    auc_ci_lower, auc_ci_upper = confidence_interval(hybrid_auc_distribution)
    print(f"Significance vs {best_baseline_name_auc} (AUC-ROC): t={t_stat_auc:.4f}, p={p_value_auc:.6f}")

    # bonferroni correction for testing two metrics (F1 and AUC-ROC) at alpha=0.05.
    corrected_alpha, (is_f1_significant, is_auc_significant) = bonferroni_correct(
        [p_value_f1, p_value_auc],
        alpha=0.05,
    )
    print(f"Bonferroni-corrected alpha: {corrected_alpha:.4f} | "
        f"F1 Significant: {is_f1_significant} | "
        f"AUC-ROC Significant: {is_auc_significant}")

    return {
        "best_baseline_name": best_baseline_name,
        "t_stat": t_stat_f1,
        "p_value": p_value_f1,
        "best_baseline_name_auc": best_baseline_name_auc,
        "t_stat_auc": t_stat_auc,
        "p_value_auc": p_value_auc,
        "auc_ci_low": auc_ci_lower,
        "auc_ci_high": auc_ci_upper,
        "corrected_alpha": corrected_alpha,
        "f1_significant": is_f1_significant,
        "auc_significant": is_auc_significant}


#compiles metrics, grid search logs, and significance test results into one dictionary.
def assemble_results(
    grid_search_results: tuple,
    hybrid_results: dict,
    baseline_results: dict,
    significance_results: dict):
    grid_weight, grid_threshold, grid_val_f1, weight_score_history = grid_search_results

    return {
        "hybrid_report": hybrid_results["hybrid_report"],
        "manual_grid_search_reference_only": {
            "note": "Reference only: manual linear blend grid-search. Not used for primary hybrid evaluation.",
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
def save_results(results_dictionary: dict, hybrid_f1_distribution: np.ndarray):
    # write formatted JSON report to disk.
    with open(RESULTS_DIR / "hybrid_fusion_report.json", "w") as report_file:
        json.dump(results_dictionary, report_file, indent=2)

    #saves bootstrap array so significance tests can be rerun without retraining
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