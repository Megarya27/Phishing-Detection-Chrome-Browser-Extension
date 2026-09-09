# This file holds the global variables/constants that
# will be called upon by other programs in this directory.

from pathlib import Path

RANDOM_SEED = 1

ROOT = Path(__file__).resolve().parent.parent

UCI_ARFF = ROOT / "raw_datasets" / "UCI Phishing dataset" / "Training Dataset.arff"

EMAIL_DIR = ROOT /"raw_datasets" /"phishing email dataset"
NAZARIO_CSV = EMAIL_DIR / "Nazario.csv"
SPAMASSASSIN_CSV = EMAIL_DIR / "SpamAssasin.csv"
ENRON_CSV = EMAIL_DIR / "Enron.csv"
LING_CSV = EMAIL_DIR / "Ling.csv"
NIGERIAN_FRAUD_CSV = EMAIL_DIR / "Nigerian_Fraud.csv"
CEAS_CSV = EMAIL_DIR / "CEAS_08.csv"
PHISHING_EMAIL_COMBINED_CSV = EMAIL_DIR / "phishing_email.csv"
# NLP training set will contain more than 15,000 labelled samples
MIN_TEXT_TRAINING_SAMPLES = 15000

ADVERSARIAL_DIR = ROOT / "prepared_datasets" / "adversarial_dataset"

STRUCTURAL_FEATURES = [
    "having_IP_Address", "URL_Length", "Shortining_Service", "having_At_Symbol",
    "double_slash_redirecting", "Prefix_Suffix", "having_Sub_Domain", "SSLfinal_State",
    "Domain_registeration_length", "Favicon", "port", "HTTPS_token", "Request_URL",
    "URL_of_Anchor", "Links_in_tags", "SFH", "Submitting_to_email", "Abnormal_URL",
    "Redirect", "on_mouseover", "RightClick", "popUpWidnow", "Iframe", "age_of_domain",
    "DNSRecord", "web_traffic", "Page_Rank", "Google_Index", "Links_pointing_to_page",
    "Statistical_report",
]

DISTILBERT_MODEL_NAME = "distilbert-base-uncased"
MAX_SEQ_LEN = 256
ARTIFACTS_DIR = ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
RESULTS_DIR = ARTIFACTS_DIR / "results"
# Create directories if they don't exist
for dir_path in [ARTIFACTS_DIR, MODELS_DIR, RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

N_BOOTSTRAP = 1000

#live structural features are those that can be computed without rendering the page a link points to.
DEAD_STRUCTURAL_FEATURES = ["web_traffic", "Page_Rank", "Google_Index", "Links_pointing_to_page"]
LIVE_STRUCTURAL_FEATURES = [f for f in STRUCTURAL_FEATURES if f not in DEAD_STRUCTURAL_FEATURES]

#are those that require rendering the page a link points to, which is 
# too slow for real-time use. These are set to default values (0) in the hybrid model.
DESTINATION_DOM_FEATURES = [
    "Favicon", "Request_URL", "URL_of_Anchor", "Links_in_tags", "SFH",
    "Submitting_to_email", "on_mouseover", "RightClick", "popUpWidnow", "Iframe",
]
PRE_CLICK_STRUCTURAL_FEATURES = [
    f for f in LIVE_STRUCTURAL_FEATURES if f not in DESTINATION_DOM_FEATURES
]