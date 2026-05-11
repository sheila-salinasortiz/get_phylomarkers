#!/usr/bin/env python3

import argparse
import subprocess
import os
import sys
import matplotlib.pyplot as plt
from datetime import datetime


def log(msg, logfile):
    """Append a message to the log file (no terminal output)."""
    with open(logfile, "a") as f:
        f.write(msg + "\n")


def log_print(msg, logfile):
    """Print to terminal AND append to log file."""
    print(msg)
    with open(logfile, "a") as f:
        f.write(msg + "\n")


def run_treecluster(tree_file, threshold, logfile):
    """
    Run TreeCluster at a given threshold and return a dictionary:
    species -> cluster_id.
    """
    tmp_out = "tmp_treecluster.tsv"

    cmd = [
        "TreeCluster.py",
        "-i", tree_file,
        "-t", str(threshold),
        "-m", "max_clade",
        "-o", tmp_out
    ]

    log(f"Running TreeCluster at threshold {threshold}", logfile)
    subprocess.run(cmd, check=True)

    clusters = {}
    with open(tmp_out) as f:
        for line in f:
            species, cluster = line.strip().split("\t")
            clusters[species] = cluster

    os.remove(tmp_out)
    return clusters


def get_cluster_size(clusters, target_group):
    """
    Correct singleton logic:

    - If cluster_id == -1 → true singleton (TreeCluster semantics)
    - Otherwise, check if ANY other taxon shares the same cluster_id
      If none → true singleton
      If some → not singleton

    For multi-taxon groups, use normal cluster_id logic.
    """

    missing = [sp for sp in target_group if sp not in clusters]
    if missing:
        raise ValueError(f"Target species not found: {', '.join(missing)}")

    first = next(iter(target_group))
    target_cluster = clusters[first]

    # SINGLE TAXON CASE
    if len(target_group) == 1:
        # Case 1: TreeCluster marks it as -1 → true singleton
        if target_cluster == "-1":
            return 1, {first}

        # Case 2: Check if ANY other taxon shares the same cluster_id
        others = {sp for sp, cl in clusters.items() if cl == target_cluster and sp != first}

        if len(others) == 0:
            return 1, {first}
        else:
            return 1 + len(others), {first} | others

    # MULTI-TAXON GROUP
    members = {sp for sp, cl in clusters.items() if cl == target_cluster}
    return len(members), members


def frange(start, stop, step):
    """Float range generator."""
    while start <= stop:
        yield round(start, 6)
        start += step


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Find the threshold at which a taxon or group becomes a singleton cluster."
    )
    parser.add_argument("--tree", required=True)
    parser.add_argument("--group", required=True,
                        help="Comma-separated list of taxa, e.g. 10,11,12")
    parser.add_argument("--min", type=float, default=0.01)
    parser.add_argument("--max", type=float, default=1.0)
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument("--outdir", default="tSuspicious_output")
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Parse comma-separated taxa into a list
    args.group = [g.strip() for g in args.group.split(",")]

    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)

    # Log file
    logfile = os.path.join(args.outdir, "tSuspicious.log")
    with open(logfile, "w") as f:
        f.write(f"tSuspicious run started: {datetime.now()}\n")
        f.write("Full command executed:\n")
        f.write("  " + " ".join(sys.argv) + "\n\n")

    log("Starting analysis...", logfile)

    target_group = set(args.group)
    thresholds = []
    sizes = []

    # Track the LAST threshold where the taxon is singleton.
    last_singleton_t = None

    min_cluster_size = float("inf")
    min_cluster_members = None
    min_cluster_assignment = None
    min_cluster_t = None

    # Evaluate cluster size across thresholds
    for t in frange(args.min, args.max, args.step):
        clusters = run_treecluster(args.tree, t, logfile)
        size, members = get_cluster_size(clusters, target_group)

        thresholds.append(t)
        sizes.append(size)

        log(f"Threshold {t}: cluster size = {size}", logfile)

        # Track minimum cluster size
        if size < min_cluster_size:
            min_cluster_size = size
            min_cluster_members = members
            min_cluster_assignment = clusters.copy()
            min_cluster_t = t

        # Track last singleton threshold
        if len(target_group) == 1:
            if size == 1:
                last_singleton_t = t
        else:
            if members == target_group:
                last_singleton_t = t

    # Report results
    if last_singleton_t is None:
        log_print("The group NEVER becomes a singleton in the tested threshold range.", logfile)
        log_print(f"Minimum observed cluster size: {min_cluster_size}", logfile)
        log_print(f"Minimum observed cluster occurs at t = {min_cluster_t}", logfile)
        final_clusters = min_cluster_assignment
    else:
        log_print("Separation threshold found", logfile)
        log_print(f"The group becomes a singleton for the last time at t = {last_singleton_t}", logfile)
        final_clusters = run_treecluster(args.tree, last_singleton_t, logfile)

    # Save final cluster assignment
    final_tsv = os.path.join(args.outdir, "final_clusters.tsv")
    with open(final_tsv, "w") as f:
        f.write("SequenceName\tClusterNumber\n")  # Header
        for sp, cl in sorted(final_clusters.items()):
            if sp.strip() == "" or sp.strip().lower() == "sequencename":
                continue
            f.write(f"{sp}\t{cl}\n")

    log(f"Final cluster assignment saved to {final_tsv}", logfile)

    # Save plot
    plot_path = os.path.join(args.outdir, "cluster_plot.png")
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, sizes, marker="o")
    plt.xlabel("Threshold")
    plt.ylabel("Cluster size")
    plt.title("Cluster size vs. threshold")
    plt.grid(True)
    plt.savefig(plot_path)
    plt.close()

    log(f"Plot saved to {plot_path}", logfile)
    log("Analysis completed.", logfile)


if __name__ == "__main__":
    main()
