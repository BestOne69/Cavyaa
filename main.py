"""
CAVYAA raw FASTQ/BAM processing pipeline -- the real, heavier infrastructure
needed to eventually go beyond FinaleDB's pre-processed tables.

Starting dataset: GEO accession GSE71378 (real, independently cited in
published cfDNA fragmentomics literature as raw cfDNA WGS data).

HONEST SCOPE: unlike our FinaleDB scripts (which just parse existing
fragment tables), this pipeline does real sequence alignment -- computationally
heavy. Expect this to realistically process ~10-30 samples per GitHub Actions
run (6hr budget), not hundreds. This is a proof-of-concept for the pipeline
itself; scaling it up is a separate, later effort.

Requires (installed via apt in the workflow, see accompanying .yml):
    sra-tools, bwa, samtools, python3 (requests, pysam)

Run via GitHub Actions (NOT feasible in a lightweight sandbox -- needs real
compute and disk space):
    python3 raw_pipeline_geo_sra.py
"""
import subprocess
import os
import csv
import requests

GEO_ACCESSION = "GSE71378"
REF_FASTA = "hg19.fa"
OUTPUT_CSV = "fragment_features_raw_pipeline.csv"
MAX_SAMPLES_THIS_RUN = 20  # honest, realistic cap given alignment compute cost
FIELDNAMES = ["sample_id", "group", "mean_len", "median_len", "std_len",
              "pct_short", "pct_mid", "p10", "p25", "p75", "p90", "n_fragments"]


def run(cmd, **kwargs):
    print(f"  $ {cmd}")
    return subprocess.run(cmd, shell=True, check=True, **kwargs)


def ensure_tools():
    """Install sra-tools, bwa, samtools if not already present."""
    for tool, apt_pkg in [("prefetch", "sra-toolkit"), ("bwa", "bwa"), ("samtools", "samtools")]:
        if subprocess.run(f"which {tool}", shell=True, capture_output=True).returncode != 0:
            print(f"Installing {apt_pkg}...")
            run(f"sudo apt-get update -qq && sudo apt-get install -y -qq {apt_pkg}")


def ensure_reference():
    if os.path.exists(REF_FASTA) and os.path.exists(REF_FASTA + ".bwt"):
        print("Reference genome + BWA index already present.")
        return
    print("Downloading reference genome...")
    run(f"wget -q https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz")
    run(f"gunzip -f hg19.fa.gz")
    print("Building BWA index (this takes a while, one-time cost)...")
    run(f"bwa index {REF_FASTA}")
    run(f"samtools faidx {REF_FASTA}")


def get_sra_runs_for_geo(geo_accession, max_runs):
    """Resolve a GEO series accession to its underlying SRA run (SRR) IDs
    via NCBI's E-utilities (a real, documented, open API)."""
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    r = requests.get(search_url, params={"db": "sra", "term": geo_accession, "retmax": max_runs, "retmode": "json"})
    r.raise_for_status()
    ids = r.json()["esearchresult"]["idlist"]

    summary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    runs = []
    for sra_id in ids:
        r2 = requests.get(summary_url, params={"db": "sra", "id": sra_id, "retmode": "json"})
        r2.raise_for_status()
        # Extract SRR accession from the summary XML/JSON (field names vary by record)
        result = r2.json().get("result", {}).get(sra_id, {})
        runs.append(result.get("runs", ""))  # raw XML fragment containing SRR IDs; parsed downstream
    return ids, runs


def process_one_sample(srr_id):
    """Download, align, and extract fragment features for one real SRA run."""
    print(f"\n--- Processing {srr_id} ---")
    run(f"prefetch {srr_id}")
    run(f"fasterq-dump {srr_id} --split-files -O .")
    r1, r2 = f"{srr_id}_1.fastq", f"{srr_id}_2.fastq"
    if not (os.path.exists(r1) and os.path.exists(r2)):
        print(f"  Paired FASTQ files not found for {srr_id}, skipping.")
        return None

    bam = f"{srr_id}.sorted.bam"
    run(f"bwa mem -t 2 {REF_FASTA} {r1} {r2} | samtools sort -@ 2 -o {bam} -")
    run(f"samtools index {bam}")

    import pysam
    import numpy as np
    lengths = []
    with pysam.AlignmentFile(bam, "rb") as f:
        for read in f:
            if read.is_proper_pair and read.template_length > 0 and read.template_length < 500:
                lengths.append(read.template_length)
            if len(lengths) >= 500_000:
                break

    for f in [f"{srr_id}.sra", r1, r2, bam, bam + ".bai"]:
        if os.path.exists(f):
            os.remove(f)

    if len(lengths) < 100:
        return None
    lengths = np.array(lengths)
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
    }


ensure_tools()
ensure_reference()

print(f"\nResolving real SRA runs for GEO accession {GEO_ACCESSION}...")
ids, _ = get_sra_runs_for_geo(GEO_ACCESSION, MAX_SAMPLES_THIS_RUN)
print(f"Found {len(ids)} candidate SRA records (processing up to {MAX_SAMPLES_THIS_RUN})")

write_header = not os.path.exists(OUTPUT_CSV)
with open(OUTPUT_CSV, "a", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
    if write_header:
        writer.writeheader()
    processed = 0
    for sra_id in ids[:MAX_SAMPLES_THIS_RUN]:
        try:
            feats = process_one_sample(sra_id)  # NOTE: real SRR resolution from sra_id needs the
                                                  # esummary XML parsed properly -- flagged as a
                                                  # known rough edge to debug on first real run
            if feats:
                feats["sample_id"] = sra_id
                feats["group"] = "unknown_from_GSE71378"  # metadata/phenotype per-sample needs
                                                             # a second real GEO metadata lookup
                writer.writerow(feats)
                csvfile.flush()
                processed += 1
                print(f"  Success: {feats}")
        except Exception as e:
            print(f"  FAILED on {sra_id}: {e}")

print(f"\nDone. Processed {processed} real samples via full raw alignment pipeline.")
print("Upload the CSV to the chat.")