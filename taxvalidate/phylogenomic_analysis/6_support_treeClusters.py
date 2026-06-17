#!/usr/bin/env python3
"""
Compute cluster profiles across t ∈ [t_min, t_max] and plot:
- Δ_mean(h)   = mean_within(h)  / mean_between(h)
- Δ_median(h) = median_within(h)/ median_between(h)

Similarity matrices (AAI/AF/POCP/ML) are loaded and relabeled as '1'..'N'
to match the tree leaf names exactly.

Two modes:
A) Default mode (no genus limits): behaves exactly like the original script.
B) Genus-limit mode: if taxonomy + labels + fusion/fragment limits are provided,
   each threshold t is evaluated for genus fragmentation/fusion and marked as valid/invalid.
"""

import argparse
import os
import subprocess
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Bio import Phylo
import re
import sys

# ---------------------------------------------------------
# Logging helper
# ---------------------------------------------------------
def make_logger(path):
    log = open(path, "w")
    def logprint(*a):
        print(*a)
        print(*a, file=log)
    return logprint, log


# ---------------------------------------------------------
# Load AAI/AF/POCP matrix with forced labels 1..N
# ---------------------------------------------------------
def load_matrix(path: str) -> pd.DataFrame:
    """
    Load a similarity matrix (AAI/AF/POCP) and force labels to '1'..'N'
    so they match the tree leaf names.
    """
    df = pd.read_csv(path, sep=r"\s+|\t+", engine="python")

    if not np.issubdtype(df.iloc[:, 0].dtype, np.number):
        df = df.drop(df.columns[0], axis=1)

    df = df.apply(pd.to_numeric, errors="coerce")

    n = df.shape[0]
    labels = [str(i + 1) for i in range(n)]
    df.index = labels
    df.columns = labels

    return df


# ---------------------------------------------------------
# Load ML distance matrix and convert to similarity
# ---------------------------------------------------------
def load_mldist_as_similarity(path: str) -> pd.DataFrame:
    """
    Load PHYLIP ML distance matrix and convert to similarity:
    similarity = 1 / (1 + distance)
    This preserves biological ordering: small distances → high similarity.
    """
    with open(path) as f:
        lines = f.read().strip().splitlines()

    n = int(lines[0].strip())
    matrix = []

    for line in lines[1:1 + n]:
        parts = line.split()
        values = list(map(float, parts[1:]))
        matrix.append(values)

    df = pd.DataFrame(matrix)

    labels = [str(i + 1) for i in range(n)]
    df.index = labels
    df.columns = labels

    df = 1 / (1 + df)
    return df


# ---------------------------------------------------------
# Taxonomy and tree labels (for genus-limit mode)
# ---------------------------------------------------------
def load_taxonomy_csv(filename):
    """
    Load Taxonomy.csv mapping species → genus.
    """
    df = pd.read_csv(filename, dtype=str).fillna("")
    mapping = {}

    for _, row in df.iterrows():
        species = row["BACTERIA"].strip()
        species = " ".join([w.rstrip(".") for w in species.split()])
        genus = row["GENUS"].strip().rstrip(".")
        mapping[species] = genus

    return mapping


def load_tree_labels(filename):
    """
    Load tree_labels.list mapping tree leaf IDs → species names.
    Species name is extracted from the [...] block.
    """
    mapping = {}
    pattern = re.compile(r"\[(.*?)\]")

    with open(filename) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue

            tree_id = parts[0].strip()
            full_label = parts[1].strip()

            m = pattern.search(full_label)
            if not m:
                continue

            content = m.group(1).replace("_", " ").strip()
            words = content.split()
            clean = " ".join(words[:2]) if len(words) >= 2 else content
            clean = " ".join([w.rstrip(".") for w in clean.split()])
            mapping[tree_id] = clean

    return mapping


def normalize_singletons_for_log(clusters):
    """
    Convert each -1 into a unique cluster ID so that singletons
    are not all grouped together.
    """
    new_clusters = {}
    counter = 1
    for sp, cl in clusters.items():
        if str(cl) == "-1":
            new_clusters[sp] = f"singleton_{counter}"
            counter += 1
        else:
            new_clusters[sp] = str(cl)
    return new_clusters


def evaluate_threshold(clusters, tree_to_species, species_to_genus):
    """
    Evaluate fragmentation and fusion for a given clustering.
    """
    cluster_to_genera = defaultdict(list)
    genus_to_clusters = defaultdict(set)

    for tree_label, cl in clusters.items():
        species = tree_to_species.get(tree_label, "UNKNOWN")
        genus = species_to_genus.get(species, "UNKNOWN")

        cluster_to_genera[cl].append(genus)
        genus_to_clusters[genus].add(cl)

    fragmented_genera = [g for g, cls in genus_to_clusters.items() if len(cls) > 1]
    fused_clusters = [cl for cl, genera in cluster_to_genera.items() if len(set(genera)) > 1]

    fused_genera = set()
    for cl in fused_clusters:
        fused_genera.update(set(cluster_to_genera[cl]))

    return (
        len(fragmented_genera),
        len(fused_clusters),
        fragmented_genera,
        list(fused_genera),
        genus_to_clusters
    )


def check_fragment_limit(genus_to_clusters, fragment_list):
    """
    Each genus in fragment_list must appear in exactly one cluster.
    """
    if not fragment_list:
        return True

    for g in fragment_list:
        if g not in genus_to_clusters:
            return False
        if len(genus_to_clusters[g]) != 1:
            return False

    return True


def check_fusion_limit(genus_to_clusters, fusion_list):
    """
    Genera in fusion_list must not share any cluster.
    """
    if not fusion_list or len(fusion_list) < 2:
        return True

    genus_clusters = {g: genus_to_clusters.get(g, set()) for g in fusion_list}

    for i in range(len(fusion_list)):
        for j in range(i + 1, len(fusion_list)):
            g1, g2 = fusion_list[i], fusion_list[j]
            if genus_clusters[g1] & genus_clusters[g2]:
                return False

    return True


# ---------------------------------------------------------
# Run TreeCluster
# ---------------------------------------------------------
def run_treecluster(tree_file, threshold):
    tmp_out = f"tmp_treecluster_{threshold:.4f}.tsv"
    cmd = [
        "Treecluster.py",
        "-i", tree_file,
        "-t", str(threshold),
        "-m", "max_clade",
        "-o", tmp_out
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    clusters = {}
    with open(tmp_out) as f:
        next(f)
        for line in f:
            leaf, cl = line.strip().split("\t")
            clusters[leaf] = int(cl)

    os.remove(tmp_out)
    return clusters


# ---------------------------------------------------------
# Compute depths and cut height h
# ---------------------------------------------------------
def compute_depths(tree):
    depths = {}
    def walk(clade, depth):
        depths[clade] = depth
        for child in clade.clades:
            bl = child.branch_length if child.branch_length else 0.0
            walk(child, depth + bl)
    walk(tree.root, 0.0)
    return depths


def clusters_from_cut(tree, depths, h):
    cut_roots = set()

    def mark(clade):
        for child in clade.clades:
            if depths[clade] < h <= depths[child]:
                cut_roots.add(child)
            mark(child)
    mark(tree.root)

    if not cut_roots:
        cut_roots = {tree.root}

    leaf_to_group = {}
    group_id = 0

    for root in cut_roots:
        group_id += 1
        for leaf in root.get_terminals():
            leaf_to_group[leaf.name] = group_id

    for leaf in tree.get_terminals():
        if leaf.name not in leaf_to_group:
            group_id += 1
            leaf_to_group[leaf.name] = group_id

    return leaf_to_group


def matches_partition(leaf_to_group, leaf_to_cluster):
    cluster_to_group = {}

    for leaf, cl in leaf_to_cluster.items():
        g = leaf_to_group[leaf]
        if cl > 0:
            if cl not in cluster_to_group:
                cluster_to_group[cl] = g
            else:
                if cluster_to_group[cl] != g:
                    return False

    group_counts = Counter(leaf_to_group.values())
    for leaf, cl in leaf_to_cluster.items():
        if cl == -1:
            if group_counts[leaf_to_group[leaf]] != 1:
                return False

    return True


def find_cut_height(tree, leaf_to_cluster):
    depths = compute_depths(tree)
    unique_depths = sorted(set(depths.values()))
    candidates = [(d1 + d2) / 2.0 for d1, d2 in zip(unique_depths[:-1], unique_depths[1:])]

    for h in candidates:
        leaf_to_group = clusters_from_cut(tree, depths, h)
        if matches_partition(leaf_to_group, leaf_to_cluster):
            return h

    return None


# ---------------------------------------------------------
# Compute mean/median within and between clusters
# ---------------------------------------------------------
def mean_median_within_between(matrix, clusters, logprint):
    leaves = [l for l in matrix.index if l in clusters]

    logprint(f"  leaves in clusters: {len(clusters)}")
    logprint(f"  leaves in matrix:   {len(matrix.index)}")
    logprint(f"  overlap leaves:     {len(leaves)}")

    if len(leaves) < 2:
        logprint("  WARNING: <2 overlapping leaves → no stats")
        return {
            "mean_within": np.nan,
            "median_within": np.nan,
            "mean_between": np.nan,
            "median_between": np.nan,
        }

    within_vals = []
    between_vals = []

    for i, li in enumerate(leaves):
        for lj in leaves[i+1:]:
            ci = clusters[li]
            cj = clusters[lj]
            val = matrix.loc[li, lj]

            if np.isnan(val):
                continue

            if ci == cj and ci != -1:
                within_vals.append(val)
            elif ci != cj:
                between_vals.append(val)

    return {
        "mean_within": np.nanmean(within_vals) if within_vals else np.nan,
        "median_within": np.nanmedian(within_vals) if within_vals else np.nan,
        "mean_between": np.nanmean(between_vals) if between_vals else np.nan,
        "median_between": np.nanmedian(between_vals) if between_vals else np.nan,
    }


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", required=True)
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--matrix_type", required=True,
                        choices=["AAI","AF","POCP","ML"])
    parser.add_argument("--t_min", type=float, default=0.0)
    parser.add_argument("--t_max", type=float, default=1.0)
    parser.add_argument("--t_step", type=float, default=0.02)
    parser.add_argument("--outdir", default="h_profiles")

    # Optional genus-limit mode
    parser.add_argument("--taxonomy", type=str,
                        help="Taxonomy.csv mapping species → genus")
    parser.add_argument("--labels", type=str,
                        help="tree_labels.list mapping tree IDs → species")
    parser.add_argument("--fusion_limit", type=str, default=None,
                        help="Comma-separated genera that must NOT share a cluster")
    parser.add_argument("--fragment_limit", type=str, default=None,
                        help="Comma-separated genera that must each appear in exactly one cluster")

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    logprint, logfile = make_logger(os.path.join(args.outdir, "run.log"))

    logprint("Command executed:")
    logprint("python", " ".join(sys.argv))
    logprint("")
    
    tree = Phylo.read(args.tree, "newick")

    # Load matrix
    if args.matrix_type == "ML":
        matrix = load_mldist_as_similarity(args.matrix)
    else:
        matrix = load_matrix(args.matrix)

    # Determine mode
    use_limits = bool(args.fusion_limit or args.fragment_limit)

    if use_limits:
        if not args.taxonomy or not args.labels:
            raise ValueError("Genus-limit mode requires --taxonomy and --labels.")
        species_to_genus = load_taxonomy_csv(args.taxonomy)
        tree_to_species = load_tree_labels(args.labels)
        fusion_limit = args.fusion_limit.split(",") if args.fusion_limit else []
        fragment_limit = args.fragment_limit.split(",") if args.fragment_limit else []
    else:
        species_to_genus = None
        tree_to_species = None
        fusion_limit = []
        fragment_limit = []

    results = []

    t = args.t_min
    while t <= args.t_max + 1e-9:
        t_round = round(t, 4)
        logprint(f"\nProcessing t = {t_round}")

        clusters_raw = run_treecluster(args.tree, t_round)
        clusters = clusters_raw.copy()

        counts = Counter(clusters.values())
        logprint("  #clusters:", len(counts))
        logprint("  cluster sizes:", dict(counts))

        h = find_cut_height(tree, clusters)
        logprint("  h found:", h)

        stats = mean_median_within_between(matrix, clusters, logprint)

        # Genus-limit evaluation (Mode B)
        if use_limits:
            clusters_eval = normalize_singletons_for_log({leaf: cl for leaf, cl in clusters.items()})
            frags, fus, frag_genera, fus_genera, genus_to_clusters = evaluate_threshold(
                clusters_eval, tree_to_species, species_to_genus
            )

            fusion_ok = check_fusion_limit(genus_to_clusters, fusion_limit)
            fragment_ok = check_fragment_limit(genus_to_clusters, fragment_limit)
            valid_limits = fusion_ok and fragment_ok

            logprint("  fragmentations:", frags, "fusions:", fus)
            logprint("  fusion_ok:", fusion_ok, "fragment_ok:", fragment_ok)
        else:
            frags = np.nan
            fus = np.nan
            fusion_ok = True
            fragment_ok = True
            valid_limits = True

        results.append({
            "t": t_round,
            "h": h,
            "mean_within": stats["mean_within"],
            "median_within": stats["median_within"],
            "mean_between": stats["mean_between"],
            "median_between": stats["median_between"],
            "fragmentations": frags,
            "fusions": fus,
            "fusion_ok": fusion_ok,
            "fragment_ok": fragment_ok,
            "valid_limits": valid_limits
        })

        t += args.t_step

    df = pd.DataFrame(results)
    df["delta_mean"] = df["mean_within"] / df["mean_between"].replace(0, np.nan)
    df["delta_median"] = df["median_within"] / df["median_between"].replace(0, np.nan)
    
    df.to_csv(os.path.join(args.outdir, "h_profile_stats.tsv"), sep="\t", index=False)

    # --- Excel-friendly copy (df stays numeric for plotting) ---
    df_excel = df.copy()

    numeric_cols = ["mean_within", "median_within", "mean_between",
                    "median_between", "delta_mean", "delta_median"]

    # Convert to floats
    for col in numeric_cols:
        df_excel[col] = pd.to_numeric(df_excel[col], errors="coerce")

    # Format using comma as decimal separator (Excel ES compatible)
    for col in numeric_cols:
        df_excel[col] = df_excel[col].apply(
            lambda x: f"{x:.6f}".replace('.', ',') if pd.notnull(x) else ""
        )

    # Save Excel-friendly version
    df_excel.to_csv(os.path.join(args.outdir, "h_profile_stats_excel.txt"),
                    sep="\t", index=False)

    # ---------------------------------------------------------
    # Plotting
    # ---------------------------------------------------------
    if not use_limits:
        # Mode A: original behavior
        plt.figure(figsize=(10,6))
        plt.plot(df["h"], df["delta_mean"], "-o", label="Δ_mean(h)", color="#1b9e77")
        plt.plot(df["h"], df["delta_median"], "-o", label="Δ_median(h)", color="#d95f02")
        plt.xlabel("Cut height h")
        plt.ylabel(f"Δ similarity ({args.matrix_type})")
        plt.title(f"Cluster separation Δ(h) across tree cuts ({args.matrix_type})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, "delta_vs_h.png"), dpi=300)
        plt.close()
    else:
        # Mode B: color points by genus-limit validity
        plt.figure(figsize=(10,6))

        valid_mask = df["valid_limits"]
        invalid_mask = ~df["valid_limits"]

        plt.scatter(df.loc[valid_mask, "h"], df.loc[valid_mask, "delta_mean"],
                    color="green", label="Valid thresholds", s=60)
        plt.scatter(df.loc[invalid_mask, "h"], df.loc[invalid_mask, "delta_mean"],
                    color="red", label="Invalid thresholds", s=60)

        plt.xlabel("Cut height h")
        plt.ylabel(f"Δ_mean ({args.matrix_type})")
        plt.title("Cluster separation Δ_mean(h) with genus-limit validation")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, "delta_mean_validity.png"), dpi=300)
        plt.close()

    logfile.close()
    print(f"Done. Results saved in {args.outdir}")


if __name__ == "__main__":
    main()