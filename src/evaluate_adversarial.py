

#This script evaluates trained models against adversarial datasets to assess the robustness
#It calculates recall scores for each model on different adversarial attacks
#providing insights into how well the models can detect phishing attempts that have been modified to evade detection.
#all rows are known phishing attacks.
#limitation:  the structural adversarial sets pair UCI URL features with real phishing email text, 
# but pairing isnt from same original attack as the UCI dataset has no text of its own.
#the semantic model still sees genuine phishing language, just not the exact email that URL came from.
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score
from transformers import DistilBertTokenizerFast, Trainer
from xgboost import XGBClassifier
from global_config import ADVERSARIAL_DIR, MODELS_DIR, PRE_CLICK_STRUCTURAL_FEATURES, RESULTS_DIR, STRUCTURAL_FEATURES
from data_text import clean_email_text
from train_baseline_distilbert import predict_proba
from train_hybrid_fusion import load_distilbert_trainer

STRUCTURAL_ADVERSARIAL_FILES = {
    "ssl_certificate_injection": ADVERSARIAL_DIR / "adv_ssl_certificate_injection.csv",
    "domain_reputation_laundering": ADVERSARIAL_DIR / "adv_domain_reputation_laundering.csv",
    "lexical_feature_normalisation": ADVERSARIAL_DIR / "adv_lexical_feature_normalisation.csv"}
SEMANTIC_ADVERSARIAL_FILE = ADVERSARIAL_DIR / "adv_semantic_obfuscation.csv"

def evaluate_structural_adversarial(
    baseline_xgb_model: XGBClassifier,
    domain_structural_model: XGBClassifier,
    meta_classifier: joblib.load,
    decision_threshold: float,
    trainer: Trainer,
    tokenizer: DistilBertTokenizerFast,
) -> dict:
    results = {}

    for attack_name, file_path in STRUCTURAL_ADVERSARIAL_FILES.items():
        attack_df = pd.read_csv(file_path)
        #baseline model uses all 30 UCI structural features, while the hybrid structural model uses only the pre-click subset of features
        #as such, slice dataframe to provide each model with the features it was trained on
        baseline_features = attack_df[STRUCTURAL_FEATURES].astype(float)
        hybrid_structural_features = attack_df[PRE_CLICK_STRUCTURAL_FEATURES].astype(float)
        labels = np.ones(len(attack_df), dtype=int)  #every row here is a known phishing sample
        baseline_probs = baseline_xgb_model.predict_proba(baseline_features)[:, 1]
        baseline_preds = (baseline_probs >= 0.5).astype(int)
        hybrid_structural_probs = domain_structural_model.predict_proba(hybrid_structural_features)[:, 1]
        cleaned_texts = [clean_email_text("", raw_text) for raw_text in attack_df["text"]] #clean email text for semantic analysis  
        semantic_probs = predict_proba(trainer, tokenizer, cleaned_texts)

        fusion_features = np.column_stack([hybrid_structural_probs, semantic_probs])
        fused_probs = meta_classifier.predict_proba(fusion_features)[:, 1]
        hybrid_preds = (fused_probs >= decision_threshold).astype(int)
        semantic_preds = (semantic_probs >= 0.5).astype(int)  #check if text alone is enough to detect phishing attempt

        results[attack_name] = {
            "n_samples": len(attack_df),
            "xgboost_recall": recall_score(labels, baseline_preds),
            "distilbert_recall": recall_score(labels, semantic_preds),
            "hybrid_recall": recall_score(labels, hybrid_preds)}
    return results

def evaluate_semantic_adversarial(trainer: Trainer, tokenizer: DistilBertTokenizerFast) -> dict:
    # no paired URLs needed for this technique as it only modifies the email text to evade detection
    attack_df = pd.read_csv(SEMANTIC_ADVERSARIAL_FILE)
    cleaned_texts = [clean_email_text("", raw_text) for raw_text in attack_df["obfuscated_text"]]
    labels = np.ones(len(attack_df), dtype=int)

    semantic_probs = predict_proba(trainer, tokenizer, cleaned_texts)
    semantic_preds = (semantic_probs >= 0.5).astype(int)

    return {
        "semantic_obfuscation": {
            "n_samples": len(attack_df),
            "distilbert_recall": recall_score(labels, semantic_preds)} }

def main():
    baseline_xgb_model = joblib.load(MODELS_DIR / "baseline_xgboost.joblib")
    domain_structural_model = joblib.load(MODELS_DIR / "structural_pre_click_paired.joblib")
    trainer, tokenizer = load_distilbert_trainer(MODELS_DIR / "distilbert_semantic" / "final")
    meta_classifier = joblib.load(MODELS_DIR / "hybrid_meta_fusion.joblib")

    with open(RESULTS_DIR / "hybrid_fusion_report.json") as f:
        hybrid_report = json.load(f)
    decision_threshold = hybrid_report["hybrid_report"]["decision_threshold"]

    results = evaluate_structural_adversarial(baseline_xgb_model, domain_structural_model, meta_classifier, decision_threshold, trainer, tokenizer)
    results.update(evaluate_semantic_adversarial(trainer, tokenizer))

    print(json.dumps(results, indent=2))
    with open(RESULTS_DIR / "adversarial_evaluation_report.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
