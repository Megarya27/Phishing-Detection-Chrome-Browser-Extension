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