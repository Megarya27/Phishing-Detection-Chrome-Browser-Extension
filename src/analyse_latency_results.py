#Summarises extensions latency log against the <2.0s target from paper (Nah, 2004).
#extension records timing breakdown for every link and keeps in
#chrome.storage.local automatically. 
#to collect benchmark: load extension, scan through batch of test emails in Gmail/Outlook, then open popup and click
#"Export latency log". 
#Then run this script using the collected logs as input. Output is JSON report in results directory

import argparse
import json
import statistics
from pathlib import Path
import numpy as np
from global_config import N_BOOTSTRAP, RANDOM_SEED, RESULTS_DIR
TARGET_LATENCY_MS = 2000  # Nah (2004): web users patience for a simple task 

# expected keys from timing dictionary logged by background worker
STAGE_NAMES = [
    "dom_scrape_and_feature_extraction_ms",
    "message_passing_ms",
    "api_round_trip_ms",
    "backend_processing_ms",
    "badge_update_ms"]

#load the JSON straight from popup's export functionality
def load_latency_records(log_path: Path) -> list[dict]:
    with open(log_path, encoding="utf-8") as f:
        return json.load(f)

#same approach as metrics.py, applied to median latency
def bootstrap_median_ci(values: list[float], n_bootstrap=N_BOOTSTRAP, seed=RANDOM_SEED, alpha=0.05):
    rng = np.random.default_rng(seed)
    arr = np.asarray(values)
    medians = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = arr[rng.integers(0, len(arr), size=len(arr))]
        medians[i] = np.median(sample)
    lower = float(np.percentile(medians, 100 * alpha / 2))
    upper = float(np.percentile(medians, 100 * (1 - alpha / 2)))
    return lower, upper

#return standard descriptive stats over distribution of timings
def summarise(values: list[float]) -> dict:
    sorted_values = sorted(values)
    p95_index = int(0.95 * (len(sorted_values) - 1))
    ci_low, ci_high = bootstrap_median_ci(values)
    return {
        "n": len(values),
        "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values),
        "median_ms_95ci": [ci_low, ci_high],
        "p95_ms": sorted_values[p95_index],
        "min_ms": min(values),
        "max_ms": max(values)}

def main():
    parser = argparse.ArgumentParser(description="Summarise latency log exported from extension popup.")
    parser.add_argument("log_path", type=Path, help="JSON file downloaded via popup's 'Export latency log' button")
    args = parser.parse_args()
    records = load_latency_records(args.log_path)
    if not records:
        raise SystemExit(f"No latency records found in {args.log_path}")
    end_to_end_ms = [r["end_to_end_ms"] for r in records]
    
    #slice data by stage to find bottlenecks if we missed target
    stage_summaries = {}
    for stage in STAGE_NAMES:
        stage_values = [r["timings"][stage] for r in records if stage in r.get("timings", {})]
        if stage_values:
            stage_summaries[stage] = summarise(stage_values)
    end_to_end_summary = summarise(end_to_end_ms)
    
    #build final report determining pass/fail on target
    report = {
        "n_samples": len(records),
        "end_to_end": end_to_end_summary,
        "stages": stage_summaries,
        "target_ms": TARGET_LATENCY_MS,
        "meets_target": end_to_end_summary["median_ms"] < TARGET_LATENCY_MS}
    print(json.dumps(report, indent=2))
    with open(RESULTS_DIR / "latency_benchmark_report.json", "w") as f:
        json.dump(report, f, indent=2)

if __name__ == "__main__":
    main()
