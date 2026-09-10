#This script builds dataset of (email text, URL, label) pairs for training and evaluating hybrid model
#through train_hybrid_fusion.py. It extracts single URL from each email, computes its structural features. 
#only uses email text is not part of DistilBERTs train or val splits to ensure validity of hybrid evaluation. 
#Limitation: 10 destination DOM features (Favicon, Request_URL, URL_of_Anchor,
#Links_in_tags, SFH, Submitting_to_email, on_mouseover, RightClick, popUpWidnow, Iframe)
# need target page loaded. as such they are not included in this dataset.

import csv
import os
import re
import socket
import sys
import time
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse
import pandas as pd
from global_config import ROOT

sys.path.insert(0, str(ROOT / "server"))
from server.feature_extraction import complete_feature_vector, STRUCTURAL_FEATURES

from data_text import load_raw_corpora, clean_email_text, build_text_dataset

OUT_DIR = ROOT / "prepared_datasets" / "paired_hybrid"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "paired_hybrid_email_urls.csv"
SESSION_PATH = OUT_DIR / "email_url_session_in_progress.csv"

URL_RE = re.compile(r"https?://[^\s\"'<>\)\]]+") #regex to find URL in email text. Stops at whitespace/common punctuation that isn't part of URL
TRAILING_PUNCT_RE = re.compile(r"[.,;:!?'\"]+$")  #regex to remove trailing punctuation from URL
JUNK_URL_SUBSTRINGS = ("unsubscribe", "click.", "track.", "beacon.", "pixel.", ".gif", ".jpg",
                       ".png", ".css", ".js", "mailto:")
MAX_URLS_PER_DOMAIN = 3  #stop same domain from coming up too many times in dataset to prevent bias.
CSV_FIELDNAMES = STRUCTURAL_FEATURES + ["url", "text", "label", "status"]

socket.setdefaulttimeout(6.0) #prevent slow WHOIS lookup from stalling

DELAY_BETWEEN_LOOKUPS_SECONDS = 0.5  #prevent being rate limited by WHOIS server

#remove trailing punctuation from URL
#filters out junk URLs
#return cleaned URL or none if invalid
def clean_url(url: str) -> str | None:
    url = TRAILING_PUNCT_RE.sub("", url)
    lowered = url.lower()
    if any(junk in lowered for junk in JUNK_URL_SUBSTRINGS):
        return None
    try:
        if not urlparse(url).netloc:
            return None
    except ValueError:
        return None
    return url

#gets registrable domain from a URL e.g. 'example.com' from "www.example.com"
def registrable_domain(url: str) -> str:
    host = urlparse(url).hostname or ""
    parts = host.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else host

#writes each row to csv directly to avoid losing all progress if the script is interrupted mid run.
class IncrementalCsvWriter:
    def __init__(self, path: Path, fieldnames: list[str]):
        self.path = path
        is_new = not path.exists()
        self._file = open(path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames, restval="")
        if is_new:
            self._writer.writeheader()
            self._file.flush()
            os.fsync(self._file.fileno()) #ensure data is written to disk immediately so progess not lost if crash occurs

    def write(self, row: dict):
        self._writer.writerow(row)
        self._file.flush()
        os.fsync(self._file.fileno()) 

    def close(self):
        self._file.close()

#gets set of all email texts that is part of DistilBERT's train or val splits
def _distilbert_seen_texts():
    splits = build_text_dataset()
    return set(splits["train"]["text"]) | set(splits["val"]["text"])


def collect_email_url_pairs():
    df = load_raw_corpora()
    excluded_texts = _distilbert_seen_texts()
    per_domain_count = defaultdict(int)
    seen_urls = set()
    pairs = []
    per_source_counts = defaultdict(int)
    n_excluded_as_seen_by_distilbert = 0

    for row in df.itertuples(index=False):  #itertuples used over iterrows as its faster on dataframes with many rows
        raw_urls = URL_RE.findall(str(row.body))

        #only keep first URL from each email that validates criteria
        chosen_url = None
        for raw_url in raw_urls:
            url = clean_url(raw_url)
            if url is None or url in seen_urls:
                continue
            domain = registrable_domain(url)
            if per_domain_count[domain] >= MAX_URLS_PER_DOMAIN:
                continue
            chosen_url = url
            break
        if chosen_url is None:
            continue

        text = clean_email_text(row.subject, row.body)
        if len(text) <= 20:
            continue
        if text in excluded_texts:
            n_excluded_as_seen_by_distilbert += 1
            continue

        seen_urls.add(chosen_url)
        per_domain_count[registrable_domain(chosen_url)] += 1
        pairs.append((chosen_url, int(row.label), text))
        per_source_counts[row.source] += 1

    print(
        f"Found {len(pairs)} usable (url, label, text) pairs across "
        f"{len(set(registrable_domain(u) for u, _, _ in pairs))} distinct domains "
        f"({n_excluded_as_seen_by_distilbert} candidates skipped as DistilBERT's "
        f"train or val split already contains exact email text"
    )
    for source, count in per_source_counts.items():
        print(f"  {source}: {count}")
    return pairs


def build_row(url: str, label: int, text: str) -> dict:
    try:
        vector = complete_feature_vector(url, {})
        row = {name: value for name, value in zip(STRUCTURAL_FEATURES, vector)}
        row.update({"url": url, "text": text, "label": label, "status": "ok"})
        return row
    except Exception as exc:
        #single failed URL shouldnt stop whole dataset building process so catch exception and log them in the row.
        return {"url": url, "text": text, "label": label, "status": f"error: {exc}"}


def main():
    if SESSION_PATH.exists():
        print(f"Found a leftover session file from previous interrupted run at "
              f"{SESSION_PATH} merging into {OUT_PATH} prior starting")
        merge_into(SESSION_PATH, OUT_PATH)
        SESSION_PATH.unlink()

    pairs = collect_email_url_pairs()

    if OUT_PATH.exists():
        import pandas as pd
        already_done = set(pd.read_csv(OUT_PATH)["url"])
        pairs = [(u, l, t) for u, l, t in pairs if u not in already_done]
        print(f"{len(already_done)} URLs already processed in {OUT_PATH}, {len(pairs)} remaining.")

    writer = IncrementalCsvWriter(SESSION_PATH, CSV_FIELDNAMES)
    try:
        for i, (url, label, text) in enumerate(pairs, 1):
            row = build_row(url, label, text)
            writer.write(row)
            print(f"[{i}/{len(pairs)}] label={label} status={row.get('status')} url={url}")
            time.sleep(DELAY_BETWEEN_LOOKUPS_SECONDS)
    finally:
        writer.close()
    merge_into(SESSION_PATH, OUT_PATH)
    SESSION_PATH.unlink()

#merge session file into output file
def merge_into(session_path: Path, out_path: Path):
    df = pd.read_csv(session_path)
    if out_path.exists():
        existing = pd.read_csv(out_path)
        before = len(existing)
        df = pd.concat([existing, df], ignore_index=True).drop_duplicates(subset="url", keep="last") #keep last so re-processed URLs newer row is used over old one
        print(f"Merged with {before} existing rows at {out_path} - {len(df)} total after removing duplicates")

    df.to_csv(out_path, index=False)
    n_ok = (df["status"] == "ok").sum()
    print(f"\nWrote {len(df)} rows to {out_path}")
    print(f"Successful feature extractions: {n_ok}/{len(df)} ({n_ok / len(df):.1%})")

if __name__ == "__main__":
    main()
