"""
Combines our two proven improvements into one pass: pulls all 9 real
cancer types + healthy (like v3), AND extracts genome-wide binned
fragmentation features across all 22 chromosomes (like our deep_extract
test) -- applying the one feature that actually improved real performance
(+0.012 AUC) to the full, wider 632+ sample dataset for the first time.

Run via GitHub Actions (same workflow pattern as before) -- this is a
bigger job than previous runs, so let it run the full ~6 hour budget if needed.
    pip install requests numpy
    python3 bulk_download_v5_genomewide.py
"""
import requests
import os
import gzip
import csv

API_BASE = "http://finaledb.research.cchmc.org/api/v1/seqrun"
DATA_BASE = "http://finaledb.research.cchmc.org/data"

MIN_FRAGS = 10_000_000
MAX_FRAGS = 60_000_000
SAMPLE_READS = 1_500_000   # per-file cap, balances depth vs total runtime across 632+ files
WINDOW = 2_000_000
CHROMS = [str(c) for c in range(1, 23)]

GROUPS = [
    ("healthy", "Healthy", 300),
    ("lung_cancer", "Lung cancer", 100),
    ("liver_cancer", "Liver cancer", 100),
    ("breast_cancer", "Breast cancer", 100),
    ("colorectal_cancer", "Colorectal cancer", 100),
    ("pancreatic_cancer", "Pancreatic cancer", 100),
    ("ovarian_cancer", "Ovarian cancer", 100),
    ("gastric_cancer", "Gastric cancer", 100),
    ("kidney_cancer", "Kidney cancer", 100),
    ("bladder_cancer", "Bladder cancer", 100),
]

OUTPUT_CSV = "fragment_features_genomewide.csv"
TMP_FILE = "_tmp_download.bgz"
FIELDNAMES = ["sample_id", "group", "mean_len", "median_len", "std_len",
              "pct_short", "pct_mid", "p10", "p25", "p75", "p90", "n_fragments",
              "genome_bin_short_frac_mean", "genome_bin_short_frac_std"]


def fetch_samples(disease, target_count, page_size=100):
    all_results, offset = [], 0
    while len(all_results) < target_count:
        params = {"disease": disease, "frag_num": f"{MIN_FRAGS},{MAX_FRAGS}",
                   "limit": page_size, "offset": offset}
        r = requests.get(API_BASE, params=params, timeout=30)
        r.raise_for_status()
        batch = r.json()["results"]
        if not batch:
            break
        all_results.extend(batch)
        offset += page_size
    return all_results


def download_to_temp(key):
    url = f"{DATA_BASE}/{key}"
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(TMP_FILE, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)


def extract_features(path, sample_n=SAMPLE_READS):
    import numpy as np
    lengths = []
    bin_short, bin_total = {}, {}
    with gzip.open(path, "rt") as f:
        for i, line in enumerate(f):
            if i % 5 != 0:
                continue
            parts = line.rstrip("\n").split("\t")
            try:
                chrom, start, end = parts[0], int(parts[1]), int(parts[2])
            except (ValueError, IndexError):
                continue
            length = end - start
            if not (0 < length < 500):
                continue
            lengths.append(length)
            if chrom in CHROMS:
                key = (chrom, start // WINDOW)
                bin_total[key] = bin_total.get(key, 0) + 1
                if length < 150:
                    bin_short[key] = bin_short.get(key, 0) + 1
            if len(lengths) >= sample_n:
                break

    lengths = np.array(lengths)
    if len(lengths) < 100:
        return None
    bin_fracs = np.array([bin_short.get(k, 0) / bin_total[k] for k in bin_total if bin_total[k] >= 15])

    return {
        "mean_len": round(float(lengths.mean()), 2),
        "median_len": round(float(np.median(lengths)), 2),
        "std_len": round(float(lengths.std()), 2),
        "pct_short": round(float((lengths < 150).mean()) * 100, 2),
        "pct_mid": round(float(((lengths >= 150) & (lengths <= 220)).mean()) * 100, 2),
        "p10": round(float(np.percentile(lengths, 10)), 2),
        "p25": round(float(np.percentile(lengths, 25)), 2),
        "p75": round(float(np.percentile(lengths, 75)), 2),
        "p90": round(float(np.percentile(lengths, 90)), 2),
        "n_fragments": len(lengths),
        "genome_bin_short_frac_mean": round(float(bin_fracs.mean()), 4) if len(bin_fracs) else None,
        "genome_bin_short_frac_std": round(float(bin_fracs.std()), 4) if len(bin_fracs) else None,
    }


done_ids = set()
write_header = not os.path.exists(OUTPUT_CSV)
if not write_header:
    with open(OUTPUT_CSV, "r") as f:
        for row in csv.DictReader(f):
            done_ids.add(row["sample_id"])

total_saved = len(done_ids)
with open(OUTPUT_CSV, "a", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
    if write_header:
        writer.writeheader()

    for group_name, disease, target in GROUPS:
        print(f"\n=== {disease} (target {target}) ===")
        samples = fetch_samples(disease, target_count=target)
        print(f"Found {len(samples)} real candidates within size range")

        processed = 0
        for s in samples:
            if processed >= target:
                break
            sid = str(s["id"])
            if sid in done_ids:
                processed += 1
                continue
            hg19_files = s.get("analysis", {}).get("hg19", [])
            frag_file = next((f for f in hg19_files if f["desc"] == "fragment"), None)
            if not frag_file:
                continue
            try:
                download_to_temp(frag_file["key"])
                feats = extract_features(TMP_FILE)
                os.remove(TMP_FILE)
                if feats is None:
                    continue
                feats["sample_id"] = sid
                feats["group"] = group_name
                writer.writerow(feats)
                csvfile.flush()
                processed += 1
                total_saved += 1
                print(f"  [{processed}/{target}] {sid} (total so far: {total_saved})")
            except Exception as e:
                print(f"  FAILED on {sid}: {e}")
                if os.path.exists(TMP_FILE):
                    os.remove(TMP_FILE)

print(f"\nDone. Total real samples with genome-wide features: {total_saved}")
print(f"Upload '{OUTPUT_CSV}' to the chat.")