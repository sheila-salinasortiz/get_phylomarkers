#!/usr/bin/env python3

import re
import json
import subprocess
import csv
import argparse
import os

ERROR_LOG = None

TAX_LEVELS = [
    "domain",
    "kingdom",
    "phylum",
    "class",
    "order",
    "family",
    "genus",
    "species"
]

HEADER_NAMES = [
    "DOMAIN", "KINGDOM", "PHYLUM", "CLASS",
    "ORDER", "FAMILY", "GENUS", "SPECIES"
]


def log_error(msg):
    with open(ERROR_LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def load_bacteria_from_txt(filename):
    """
    Rule: ALWAYS take the first 2 words inside brackets,
    replacing '_' with a space.
    Examples:
      [Antrihabitans_sp._YC3-6] → Antrihabitans sp.
      [Jongsikchunia_kroppenstedtii_DSM_45133] → Jongsikchunia kroppenstedtii
      [Mycobacterium_avium_subsp._avium] → Mycobacterium avium
    """
    pattern = re.compile(r"\[(.*?)\]")
    bacteria = []

    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            m = pattern.search(line)
            if not m:
                bacteria.append(line.strip())
                continue

            content = m.group(1).replace("_", " ").strip()
            parts = content.split()

            if len(parts) >= 2:
                name = " ".join(parts[:2])
            else:
                name = content

            bacteria.append(name.strip())

    return bacteria


def validate_name(name):
    if not name or name.strip() == "":
        return False
    parts = name.split()
    if not parts[0][0].isupper():
        return False
    return True


def query_ncbi_datasets(name):
    """
    Executes NCBI Datasets EXACTLY as in the terminal:
    datasets summary taxonomy taxon "Name"
    """
    try:
        cmd = f'datasets summary taxonomy taxon "{name}"'
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True
        )

        if result.returncode != 0 or not result.stdout.strip():
            return None

        return json.loads(result.stdout)

    except Exception as e:
        log_error(f"{name} → Error running datasets: {str(e)}")
        return None


def extraer_taxonomia(data):
    """
    Extracts DOMAIN → SPECIES from the REAL datasets output.
    """
    lineage = {lvl: "" for lvl in TAX_LEVELS}

    try:
        classification = data["reports"][0]["taxonomy"]["classification"]
    except KeyError:
        return lineage

    for lvl in TAX_LEVELS:
        if lvl in classification and "name" in classification[lvl]:
            lineage[lvl] = classification[lvl]["name"]

    return lineage


def process(bacterias, output):
    results = []

    for b in bacterias:
        print(f"Processing: {b}")

        if not validate_name(b):
            log_error(f"{b} → name no valid")
            results.append([b] + [""] * len(TAX_LEVELS))
            continue

        data = query_ncbi_datasets(b)
        if not data or "reports" not in data:
            log_error(f"{b} → Not found in NCBI Datasets")
            results.append([b] + [""] * len(TAX_LEVELS))
            continue

        lineage = extraer_taxonomia(data)
        row = [b] + [lineage[l] for l in TAX_LEVELS]
        results.append(row)

    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["BACTERIA"] + HEADER_NAMES)
        writer.writerows(results)

    print("\nCSV generated:", output)
    print("Error log:", ERROR_LOG)


def parse_args():
    parser = argparse.ArgumentParser(description="Extract taxonomy from NCBI Datasets.")
    parser.add_argument("--input", required=True, help="Input TXT file with labels")
    parser.add_argument("--output", default="Taxonomy.csv", help="Output CSV file")
    parser.add_argument("--errorlog", default="taxonomy_errors.log", help="Error log file")
    return parser.parse_args()


def main():
    global ERROR_LOG

    args = parse_args()

    ERROR_LOG = args.errorlog

    # Clean previous log if exists
    if os.path.exists(ERROR_LOG):
        os.remove(ERROR_LOG)

    bacterias = load_bacteria_from_txt(args.input)
    process(bacterias, args.output)


if __name__ == "__main__":
    main()