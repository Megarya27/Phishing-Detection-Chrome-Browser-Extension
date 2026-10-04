#This script implements the flask inference api for hybrid phishing detector
#it receives email text, target url and preclick features from  extension service worker
# evaluates inputs using distilBERT and XGBoost models then returns fused probability scores.
#default address is http://localhost:5000

import logging
from flask import Flask, jsonify, request
from flask_cors import CORS
from inference import HybridPhishingDetector

#configure server logging to log to console
logging.basicConfig(level=logging.INFO)
api_logger = logging.getLogger("phishing-guard-api")

app = Flask(__name__)
#enable cors so browser extension service worker can make cross-origin requests
CORS(app)

# load models once at startup to keep inference requests fast and detect missing files immediately.
api_logger.info("Loading hybrid phishing detector models...")
phishing_detector = HybridPhishingDetector()
api_logger.info("Models loaded. Decision threshold = %s", phishing_detector.decision_threshold)


#health check route to verify server status and report active decision threshold
@app.route("/health", methods=["GET"])
def check_health():
    return jsonify({
        "status": "ok",
        "decision_threshold": phishing_detector.decision_threshold})

#main prediction route to score incoming urls and email content
@app.route("/predict", methods=["POST"])
def predict_phishing():
    #silent=True prevents flask from crashing on malformed json, returning None instead
    request_data = request.get_json(silent=True) or {}
    target_url = request_data.get("url")
    email_body_text = request_data.get("text", "")
    extracted_features = request_data.get("structural_features", {})

    #target url is only mandatory input
    if not target_url:
        return jsonify({"error": "Missing required field 'url'"}), 400

    try:
        detection_result = phishing_detector.predict(target_url, email_body_text, extracted_features) #run hybrid inference model

        #log prediction breakdown, probabilities and text snippet for monitoring
        api_logger.info(
            "url=%s fused=%.4f struct=%.4f sem=%.4f api_verdict=%s@%.2f text_preview=%r",
            target_url,
            detection_result["phishing_probability"],
            detection_result["structural_probability"],
            detection_result["semantic_probability"],
            detection_result["is_phishing"],
            phishing_detector.decision_threshold,
            (email_body_text or "")[:80])
        return jsonify(detection_result)

    except Exception:
        #log full stack trace to server console but hide internal tracebacks from client
        api_logger.exception("Inference failed for url=%s", target_url)
        return jsonify({"error": "Internal inference error"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False) #bind to 0.0.0.0 to accept traffic through tunnels and external network interfaces