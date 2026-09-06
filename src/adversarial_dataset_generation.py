# This program generates the adversarial evaluation set
# It builds 4 sets of 500 samples each, using the following techniques:
# 1. SSL Certificate Injection  
# 2. Domain Reputation Laundering
# 3. Lexical Feature Normalisation
# 4. Semantic Obfuscation (email text paraphrasing)
# All samples are drawn from the UCI Phishing Websites dataset and the Nazario / phishing_email corpora, 
#which are publicly available and already labelled as phishing.

import csv
import random
import re
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from data_structural import load_structural_splits
from global_config import RANDOM_SEED, UCI_ARFF, PHISHING_EMAIL_COMBINED_CSV as EMAIL_CSV, NAZARIO_CSV, ADVERSARIAL_DIR as OUT_DIR
from data_text import build_text_dataset
random.seed(RANDOM_SEED)
OUT_DIR.mkdir(exist_ok=True) #ensure output directory exists/create if not

N_PER_TECHNIQUE = 500 # number of samples per adversarial technique
#list of structural features in the UCI Phishing Websites dataset, plus the label column
STRUCTURAL_FEATURES = [ 
    "having_IP_Address", "URL_Length", "Shortining_Service", "having_At_Symbol",
    "double_slash_redirecting", "Prefix_Suffix", "having_Sub_Domain", "SSLfinal_State",
    "Domain_registeration_length", "Favicon", "port", "HTTPS_token", "Request_URL",
    "URL_of_Anchor", "Links_in_tags", "SFH", "Submitting_to_email", "Abnormal_URL",
    "Redirect", "on_mouseover", "RightClick", "popUpWidnow", "Iframe", "age_of_domain",
    "DNSRecord", "web_traffic", "Page_Rank", "Google_Index", "Links_pointing_to_page",
    "Statistical_report"]

#load only known phishing samples from the UCI Phishing Websites dataset, where label = -1
# load_structural_splits() is used to ensure the same 70:15:15 train/validation/test split 
# is used for both structural and hybrid models
def load_uci_phishing_rows():
    splits = load_structural_splits()  # feature_columns defaults to all 30 UCI features
    X_test, y_test = splits["X_test"], splits["y_test"]
    phishing_mask = y_test == 1  #1 = phishing, 0 = legitimate
    return X_test[phishing_mask].to_dict("records")

#Randomly selects a subset of phishing rows using random seed offset to ensure no overlap in samples drawn for each technique
def sample_phishing_rows(rows, n, seed_offset):
    rng = random.Random(RANDOM_SEED + seed_offset)
    sampled_items = rng.sample(rows, n)
    
    #create clean copies of each dictionary row
    copied_rows = []
    for r in sampled_items:
        copied_rows.append(dict(r))
    return copied_rows
held_out_phishing_texts_cache = None

#real phishing email text samples held out from DistilBERT's training set
def load_held_out_phishing_texts(n, seed_offset):
    global held_out_phishing_texts_cache
    if held_out_phishing_texts_cache is None:
        #ensures validation + test sets were not used to train DistilBERT.
        splits = build_text_dataset()
        validation_texts = splits["val"]
        test_texts = splits["test"]
        held_out_texts = pd.concat([validation_texts, test_texts])
        phishing_rows = held_out_texts[held_out_texts["label"] == 1]
        held_out_phishing_texts_cache = phishing_rows["text"].tolist()
    texts = held_out_phishing_texts_cache
    rng = random.Random(RANDOM_SEED + seed_offset)

    if n <= len(texts):
        # sample() does not select the same text twice.
        return rng.sample(texts, n)

    # choice() allows same text to selected more than once.
    selected_texts = []
    for i in range(n):
        selected_texts.append(rng.choice(texts))
    return selected_texts


#Technique 1: SSL Certificate Injection
# simulates attacker purchasing legitimate SSL certificate to bypass security checks
# UCI encoding: 1 = legitimate, 0 = suspicious, -1 = phishing.
def apply_ssl_certificate_injection(row):
    row["SSLfinal_State"] = 1     # trusted/valid certificate state (legitimate)
    row["HTTPS_token"] = 1        # no "https" token stuffed into the domain string (legitimate)
    row["Favicon"] = 1            # favicon looks legitimate
    row["adversarial_technique"] = "ssl_certificate_injection"
    return row


# Technique 2: Domain Reputation Laundering 
#simulate hosting on reputable cloud platform subdomain
def apply_domain_reputation_laundering(row):
    row["Domain_registeration_length"] = 1  # appears as shared parent domain
    row["age_of_domain"] = 1                # inherits age of legitimate platform
    row["DNSRecord"] = 1                    # valid DNS record present
    row["having_Sub_Domain"] = 0            # exactly one subdomain 
    row["Abnormal_URL"] = 1                 # hostname matches WHOIS/URL, no longer flagged abnormal (legitimate)
    row["Google_Index"] = 1                 # page indexed 
    row["web_traffic"] = 1                  # inherits platform level traffic rank
    row["Page_Rank"] = 1                    # inherits platform level page rank
    row["adversarial_technique"] = "domain_reputation_laundering"
    return row


#Technique 3: Lexical Feature Normalisation
# normalise URL string statistics to match benign URL 
def apply_lexical_feature_normalisation(row):
    row["URL_Length"] = 1                # normal length (legitimate)
    row["Shortining_Service"] = 1        # no shortener (legitimate)
    row["having_At_Symbol"] = 1          # no '@' obfuscation (legitimate)
    row["Prefix_Suffix"] = 1             # no '-' prefix/suffix (legitimate)
    row["double_slash_redirecting"] = 1  # "//" not misused for redirection (legitimate)
    row["having_IP_Address"] = 1         # domain name used, not raw IP (legitimate)
    row["adversarial_technique"] = "lexical_feature_normalisation"
    return row


# Technique 4: Semantic Obfuscation (email text paraphrasing)
SYNONYM_MAP = {
    r"\burgent(ly)?\b": ["time-sensitive", "immediate", "pressing"],
    r"\bverify\b": ["confirm", "validate", "authenticate"],
    r"\baccount\b": ["profile", "membership", "user record"],
    r"\bsuspend(ed)?\b": ["restrict(ed)", "lock(ed)", "deactivat(ed)"],
    r"\bclick\b": ["select", "follow", "open"],
    r"\bpassword\b": ["passcode", "login credential", "access code"],
    r"\bsecurity\b": ["safety", "protection", "integrity"],
    r"\bconfirm\b": ["validate", "acknowledge", "re-affirm"],
    r"\bimmediately\b": ["right away", "without delay", "at once"],
    r"\bexpire[sd]?\b": ["lapse(s)", "end(s)", "become invalid"],
}

FILLER_OPENERS = [
    "As part of our routine account review, ",
    "Following a recent system check, ",
    "In line with our updated policy, ",
    "To keep your experience uninterrupted, ",
    "",
]

#helper function to pick a random synonym from the options list
# preserves case and handles optional suffixes
def _pick_synonym(match, options):
    word = match.group(0)
    repl = random.choice(options)
    if "(" in repl:
        base, suffix = re.match(r"([a-z]+)\(([a-z]*)\)", repl).groups()
        repl = base + suffix
    #keep capitalised if original word was capitalised
    if word[0].isupper():
        repl = repl.capitalize()
    return repl

#Rule based synonym substitution and sentence reordering to obfuscate email text.
def paraphrase_email_body(text):
    text = text[:2000] #only paraphrase first 2000 characters
    #replace target phishing keywords with alternative synonyms
    for pattern, options in SYNONYM_MAP.items():
        text = re.sub(pattern, lambda m: _pick_synonym(m, options), text, flags=re.IGNORECASE)
 ## Split text into sentences and shuffle middle sentences if there are enough of them
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) > 2:
        mid = sentences[1:-1]
        random.shuffle(mid)
        sentences = [sentences[0]] + mid + [sentences[-1]]
        text = " ".join(sentences)
 #add a random opener to the start of the email body to further obfuscate it
    opener = random.choice(FILLER_OPENERS)
    return opener + text

#builds collection of paraphrased email samples for the semantic test set
def build_semantic_obfuscation_set(n):
    bodies = load_held_out_phishing_texts(n, seed_offset=4)
    out = []
    for i, body in enumerate(bodies):
        out.append({
            "sample_id": f"semantic_obfuscation_{i:04d}",
            "adversarial_technique": "semantic_obfuscation",
            "original_text": body,
            "obfuscated_text": paraphrase_email_body(body),
            "label": "phishing",
        })
    return out

#writes structural modification features to a CSV file 
def write_structural_csv(path, rows, technique_name, texts):
    fieldnames = ["sample_id", "adversarial_technique"] + STRUCTURAL_FEATURES + ["text", "label"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, (row, text) in enumerate(zip(rows, texts)):
            out_row = {
                "sample_id": f"{technique_name}_{i:04d}",
                "adversarial_technique": row["adversarial_technique"],
                "text": text,
                "label": "phishing",
            }
            out_row.update({k: row[k] for k in STRUCTURAL_FEATURES})
            writer.writerow(out_row)

#writes semantic obfuscation samples to CSV file
def write_semantic_csv(path, rows):
    fieldnames = ["sample_id", "adversarial_technique", "original_text", "obfuscated_text", "label"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    phishing_rows = load_uci_phishing_rows()
    print(f"Loaded {len(phishing_rows)} known-phishing structural samples from UCI dataset")

    ssl_rows = [apply_ssl_certificate_injection(r) for r in
                sample_phishing_rows(phishing_rows, N_PER_TECHNIQUE, seed_offset=1)] #seed_offset ensure no overlap in samples drawn for each technique
    laundering_rows = [apply_domain_reputation_laundering(r) for r in
                        sample_phishing_rows(phishing_rows, N_PER_TECHNIQUE, seed_offset=2)]
    lexical_rows = [apply_lexical_feature_normalisation(r) for r in
                     sample_phishing_rows(phishing_rows, N_PER_TECHNIQUE, seed_offset=3)]
    ssl_texts = load_held_out_phishing_texts(N_PER_TECHNIQUE, seed_offset=11)
    laundering_texts = load_held_out_phishing_texts(N_PER_TECHNIQUE, seed_offset=12)
    lexical_texts = load_held_out_phishing_texts(N_PER_TECHNIQUE, seed_offset=13)
    semantic_rows = build_semantic_obfuscation_set(N_PER_TECHNIQUE)

    write_structural_csv(OUT_DIR / "adv_ssl_certificate_injection.csv", ssl_rows,
                          "ssl_certificate_injection", ssl_texts)
    write_structural_csv(OUT_DIR / "adv_domain_reputation_laundering.csv", laundering_rows,
                          "domain_reputation_laundering", laundering_texts)
    write_structural_csv(OUT_DIR / "adv_lexical_feature_normalisation.csv", lexical_rows,
                          "lexical_feature_normalisation", lexical_texts)
    write_semantic_csv(OUT_DIR / "adv_semantic_obfuscation.csv", semantic_rows)

    print(f"Wrote {N_PER_TECHNIQUE} samples per technique to {OUT_DIR}")
    print("Total adversarial evaluation set size:", 4 * N_PER_TECHNIQUE)


if __name__ == "__main__":
    main()