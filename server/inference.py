#loads trained hybrid model components at startup
#provides single predict() function used by app.py to score links and email text
#uses trained logistic regression metaclassifier to fuse structural and semantic probabilities

import json
import time
from pathlib import Path
import joblib
import numpy as np
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
from feature_extraction import complete_feature_vector
from text_cleaning import clean_email_text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "artifacts" / "models"
RESULTS_DIR = PROJECT_ROOT / "artifacts" / "results"

#fallback decision threshold if report file is missing
DEFAULT_DECISION_THRESHOLD = 0.1

#maximum token sequence length for distilBERT tokenisation
MAX_SEQUENCE_LENGTH = 256


class HybridPhishingDetector:
    def __init__(self):
        #find preclick structural model on disk with fallbacks to older version if not found
        structural_model_path = MODELS_DIR / "structural_pre_click_paired.joblib"
        if not structural_model_path.exists():
            structural_model_path = MODELS_DIR / "structural_domain_matched_paired.joblib"
        if not structural_model_path.exists():
            structural_model_path = MODELS_DIR / "baseline_xgboost.joblib"

        #load structural model and metafusion classifier into memory
        self.structural_model = joblib.load(structural_model_path)
        self.meta_classifier = joblib.load(MODELS_DIR / "hybrid_meta_fusion.joblib")

        #load tokeniser and weights from same directory to ensure token ids match
        semantic_model_directory = MODELS_DIR / "distilbert_semantic" / "final"
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(semantic_model_directory)

        #use gpu if available else run on cpu
        self.execution_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.semantic_model = DistilBertForSequenceClassification.from_pretrained(semantic_model_directory).to(self.execution_device)
        self.semantic_model.eval() #set model to evaluation mode to disable dropout during inference

        #load optimal decision cutoff determined during model validation
        self.decision_threshold = self._load_decision_threshold()

    @staticmethod
    def _load_decision_threshold() -> float:
        #read threshold directly from training report so it matches trained models
        report_path = RESULTS_DIR / "hybrid_fusion_report.json"
        if report_path.exists():
            with open(report_path) as report_file:
                report_data = json.load(report_file)
            return report_data["hybrid_report"]["decision_threshold"]
        return DEFAULT_DECISION_THRESHOLD  #return default if report file is not available

    def _calculate_structural_probability(self, target_url: str, client_features: dict) -> float:
        full_feature_vector = complete_feature_vector(target_url, client_features)  #combine client extracted url features with server lookups in exact expected order

        #predict probability using xgboost
        prediction_probabilities = self.structural_model.predict_proba([full_feature_vector])
        phishing_probability = prediction_probabilities[:, 1][0]
        return float(phishing_probability)

    def _calculate_semantic_probability(self, email_body_text: str) -> float:
        cleaned_text = clean_email_text("", email_body_text)  #clean text using same pipeline used during distilBERT training

        #return zero risk for empty text to avoid random token scoring
        if not cleaned_text or not cleaned_text.strip():
            return 0.0

        #tokenise text to tensor format with truncation and padding
        tokenized_inputs = self.tokenizer(
            cleaned_text,
            truncation=True,
            padding="max_length",
            max_length=MAX_SEQUENCE_LENGTH,
            return_tensors="pt").to(self.execution_device)

        #run inference without calculating gradients to save memory and compute time
        with torch.no_grad():
            model_outputs = self.semantic_model(**tokenized_inputs)
            output_logits = model_outputs.logits

        #apply softmax to convert logits to probabilities, selecting 1 (phishing)
        probabilities = torch.softmax(output_logits, dim=1)
        phishing_probability = probabilities[0, 1].item()
        return float(phishing_probability)

    def predict(self, target_url: str, email_body_text: str, client_features: dict) -> dict:
        #track start time using a monotonic clock for accurate latency benchmarking
        timer_start = time.perf_counter()

        #get independent probability scores from both models
        structural_probability = self._calculate_structural_probability(target_url, client_features)
        semantic_probability = self._calculate_semantic_probability(email_body_text)

        #combine scores using trained metaclassifier in [structural, semantic] order
        meta_input = [[structural_probability, semantic_probability]]
        fused_probability = float(self.meta_classifier.predict_proba(meta_input)[:, 1][0])

        #determine boolean verdict based on server decision cutoff
        is_phishing_verdict = fused_probability >= self.decision_threshold

        #calculate total backend inference time in milliseconds
        backend_processing_ms = (time.perf_counter() - timer_start) * 1000

        #return both individual component probabilities and final combined score
        return {
            "phishing_probability": fused_probability,
            "is_phishing": bool(is_phishing_verdict),
            "semantic_probability": semantic_probability,
            "structural_probability": structural_probability,
            "decision_threshold": self.decision_threshold,
            "backend_processing_ms": backend_processing_ms}