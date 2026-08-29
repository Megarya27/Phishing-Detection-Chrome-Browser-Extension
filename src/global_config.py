# This file holds the global variables/constants that
# will be called upon by other programs in this directory.

from pathlib import Path

RANDOM_SEED = 1

ROOT = Path(__file__).resolve().parent.parent

UCI_ARFF = ROOT / "raw_datasets" / "UCI Phishing dataset" / "Training Dataset.arff"

