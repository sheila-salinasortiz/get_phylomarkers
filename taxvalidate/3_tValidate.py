#!/usr/bin/env python3

import argparse
import subprocess
import os
import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
import re
from matplotlib.colors import ListedColormap


# ---------------------------------------------------------
# 1. Load taxonomy CSV (species → genus)
# ---------------------------------------------------------
def load_taxonomy_csv(filename):
    df = pd.read_csv(filename, dtype=str).fillna("")
    mapping = {}

    for _, row in df.iterrows():
        species = row["BACTERIA"].strip()
        species = " ".join([w.rstrip(".") for w in species.split()])
        genus = row["GENUS"].strip().rstrip(".")
        mapping[species] = genus

    return mapping


# ---------------------------------------------------------
# 2. Load tree labels
# ---------------------------------------------------------
def load_tree_labels(filename):
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


# ---------------------------------------------------------
# 3. Run TreeCluster
# ---------------------------------------------------------
def run_treecluster(tree_file, threshold, output_file=None):
    tmp_out = output_file if output_file else "tmp_treecluster.tsv"

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
        for line in f:
            sp, cl = line.strip().split("\t")
            clusters[sp] = cl

    if output_file is None:
        os.remove(tmp_out)

    return clusters

def normalize_singletons_for_log(clusters):
    new_clusters = {}
    counter = 1
    for sp, cl in clusters.items():
        if cl == "-1":
            new_clusters[sp] = f"singleton_{counter}"
            counter += 1
        else:
            new_clusters[sp] = cl
    return new_clusters


# ---------------------------------------------------------
# 4. Evaluate fragmentation and fusion
# ---------------------------------------------------------
def evaluate_threshold(clusters, tree_to_species, species_to_genus):
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


# ---------------------------------------------------------
# 5. Genus-based limit checks
# ---------------------------------------------------------
def check_fragment_limit(genus_to_clusters, fragment_list):
    if not fragment_list:
        return True

    for g in fragment_list:
        if g not in genus_to_clusters:
            return False
        if len(genus_to_clusters[g]) != 1:
            return False

    return True


def check_fusion_limit(genus_to_clusters, fusion_list):
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
# 6. GTDB-style matrix
# ---------------------------------------------------------
def build_gtdb_matrix(all_clusters, thresholds, tree_to_species, species_to_genus, all_genera):
    matrix = []

    for t in thresholds:
        clusters = all_clusters[t]

        # FIX: treat each -1 as an independent cluster ONLY for the GTDB plot
        fixed_clusters = {}
        counter = 1
        for sp, cl in clusters.items():
            if cl == "-1":
                fixed_clusters[sp] = f"singleton_{counter}"
                counter += 1
            else:
                fixed_clusters[sp] = cl

        cluster_to_genera = defaultdict(list)
        genus_to_clusters = defaultdict(set)

        for tree_label, cl in fixed_clusters.items():
            species = tree_to_species.get(tree_label)
            genus = species_to_genus.get(species)
            if genus is None:
                continue

            cluster_to_genera[cl].append(genus)
            genus_to_clusters[genus].add(cl)

        row = []
        for g in all_genera:
            cls = genus_to_clusters.get(g, set())

            if not cls:
                row.append(0)
                continue

            fragmented = len(cls) > 1
            fused = any(len(set(cluster_to_genera[cl])) > 1 for cl in cls)

            if fragmented and fused:
                row.append(3)
            elif fragmented:
                row.append(1)
            elif fused:
                row.append(2)
            else:
                row.append(0)

        matrix.append(row)

    return np.array(matrix, dtype=int)



# ---------------------------------------------------------
# 7. Float range
# ---------------------------------------------------------
def frange(start, stop, step):
    while start <= stop:
        yield round(start, 6)
        start += step


# ---------------------------------------------------------
# 8. Select best threshold with genus limits
# ---------------------------------------------------------
def find_best_t_by_genera(thresholds, limit_status):
    for t in thresholds:
        fusion_ok, fragment_ok = limit_status[t]
        if fusion_ok and fragment_ok:
            return t, True

    for t in thresholds:
        fusion_ok, fragment_ok = limit_status[t]
        if fusion_ok or fragment_ok:
            return t, False

    return thresholds[0], False


# ---------------------------------------------------------
# 9. Select best threshold without limits
# ---------------------------------------------------------
def compute_mixing_score_from_clusters(clusters, tree_to_species, species_to_genus):
    cluster_to_genera = defaultdict(list)

    for tree_label, cl in clusters.items():
        species = tree_to_species.get(tree_label, "UNKNOWN")
        genus = species_to_genus.get(species, "UNKNOWN")
        cluster_to_genera[cl].append(genus)

    score = 0
    for cl, genera in cluster_to_genera.items():
        if len(set(genera)) > 1:
            score += len(genera)

    return score


# ---------------------------------------------------------
# 10. AAI VALIDATION
# ---------------------------------------------------------

# Higher values of AAI implies that 2 bacterias are close in terms of evolution.
# Remember: AAI: how much similarity present the proteosomes of 2 bacterias.

def validate_t_with_AAI(best_clusters_file, aai_file, outdir):
    print("\nRunning AAI validation...")

    # Load AAI matrix
    aai_df = pd.read_csv(aai_file, sep="\t", index_col=0)

    # Clean AAI names
    def clean_name(name):
        name = re.sub(r"\.(gz|zip|bz2|xz|gbff|fna|faa|fa|txt|tab)+$", "", name)
        name = name.replace("_", " ")
        m = re.search(r"\b([A-Z][a-zA-Z]+)\s+([a-z]+)\b", name)
        if m:
            return f"{m.group(1)} {m.group(2)}"
        return name.strip()

    aai_df.index = [clean_name(x) for x in aai_df.index]
    aai_df.columns = [clean_name(x) for x in aai_df.columns]

    # ---------------------------------------------------------
    # LOAD TREE LABELS (ID → species name)
    # ---------------------------------------------------------
    tree_labels = {}
    with open("resultsOutgroupDataset/tree_labels.list") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            tid = parts[0]
            raw = parts[1].strip("[]")
            tree_labels[tid] = clean_name(raw)

    # ---------------------------------------------------------
    # LOAD FINAL CLUSTERS (IDs → cluster)
    # ---------------------------------------------------------
    final_clusters = {}
    with open(best_clusters_file) as f:
        for line in f:
            tid, cl = line.strip().split("\t")
            if tid in tree_labels:
                species = tree_labels[tid]
                final_clusters[species] = cl

    species_list = list(final_clusters.keys())

    within = []
    between = []
    within_pairs = []
    between_pairs = []


    # ---------------------------------------------------------
    # COMPARE ALL PAIRS
    # ---------------------------------------------------------
    for i in range(len(species_list)):
        for j in range(i + 1, len(species_list)):
            sp1 = species_list[i]
            sp2 = species_list[j]

            if sp1 not in aai_df.index or sp2 not in aai_df.index:
                continue

            aai = aai_df.loc[sp1, sp2]

            if final_clusters[sp1] == final_clusters[sp2]:
                within.append(aai)
                within_pairs.append((sp1, sp2, aai))
            else:
                between.append(aai)
                between_pairs.append((sp1, sp2, aai))


    # Save results
    csv_path = os.path.join(outdir, "aai_validation_pairs.csv")
    with open(csv_path, "w") as out:
        out.write("pair_type,species1,species2,AAI\n")
        for (sp1, sp2, aai) in within_pairs:
            out.write(f"within,{sp1},{sp2},{aai}\n")
        for (sp1, sp2, aai) in between_pairs:
            out.write(f"between,{sp1},{sp2},{aai}\n")

    summary_path = os.path.join(outdir, "aai_validation_summary.txt")
    with open(summary_path, "w") as out:
        out.write("AAI VALIDATION SUMMARY\n\n")
        out.write(f"Within-cluster pairs: {len(within)}\n")
        out.write(f"Between-cluster pairs: {len(between)}\n\n")

        if within:
            out.write(f"Within mean: {np.mean(within):.3f}\n")
            out.write(f"Within std:  {np.std(within):.3f}\n\n")
        if between:
            out.write(f"Between mean: {np.mean(between):.3f}\n")
            out.write(f"Between std:  {np.std(between):.3f}\n")

    # Plot
    if within or between:
        plt.figure(figsize=(10, 6))
        if within:
            plt.hist(within, bins=30, alpha=0.75, label="Within clusters", color="#e41a1c")
        if between:
            plt.hist(between, bins=30, alpha=0.75, label="Between clusters", color="#377eb8")
        plt.xlabel("AAI (%)")
        plt.ylabel("Frequency")
        plt.title("AAI Distribution: Within vs Between Clusters")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "aai_histogram.png"))
        plt.close()

    print("AAI validation completed.")


# ---------------------------------------------------------
# 11. MAIN
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="TreeCluster threshold analysis with GTDB-style visualization.")
    parser.add_argument("--tree", required=True, help="Normalized rooted tree .nwk")
    parser.add_argument("--taxonomy", required=True, help="Taxonomy.csv")
    parser.add_argument("--labels", required=True, help="tree_labels.list")
    parser.add_argument("--min", type=float, default=0.01)
    parser.add_argument("--max", type=float, default=1.0)
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument("--outdir", default="threshold_analysis")

    parser.add_argument("--fusion_limit", type=str, default=None,
                        help="This genus are not together in a single cluster.")
    parser.add_argument("--fragment_limit", type=str, default=None,
                        help="Each genus is one cluster.")

    parser.add_argument("--aai", type=str, help="AAI matrix (.tab) for validation")

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    species_to_genus = load_taxonomy_csv(args.taxonomy)
    tree_to_species = load_tree_labels(args.labels)
    all_genera = sorted(set(species_to_genus.values()))

    fusion_limit = args.fusion_limit.split(",") if args.fusion_limit else []
    fragment_limit = args.fragment_limit.split(",") if args.fragment_limit else []

    log_path = os.path.join(args.outdir, "analysis.log")
    csv_path = os.path.join(args.outdir, "threshold_summary.csv")
    details_csv = os.path.join(args.outdir, "fragmentation_fusion_details.csv")
    gtdb_plot_path = os.path.join(args.outdir, "gtdb_plot.png")
    classic_plot_path = os.path.join(args.outdir, "threshold_plot.png")

    thresholds = []
    frag_list = []
    fus_list = []
    all_clusters = {}
    limit_status = {}

    with open(log_path, "w") as log, \
         open(csv_path, "w", newline="") as csvfile, \
         open(details_csv, "w", newline="") as detfile:

        writer = csv.writer(csvfile)
        writer.writerow(["threshold", "fragmentations", "fusions"])

        det_writer = csv.writer(detfile)
        det_writer.writerow(["threshold", "fragmented_genera", "fused_genera"])

        for t in frange(args.min, args.max, args.step):
            log.write(f"Evaluating t={t}\n")

            clusters = run_treecluster(args.tree, t)
            all_clusters[t] = clusters

            # Treat -1 as independent clusters
            clusters_corrected = normalize_singletons_for_log(clusters)

            frags, fus, frag_genera, fus_genera, genus_to_clusters = evaluate_threshold(
                clusters_corrected, tree_to_species, species_to_genus
            )

            # Count real singletons (cluster == "-1")
            real_singletons = sum(1 for cl in clusters.values() if cl == "-1")

            log.write(f"  Singletons: {real_singletons}\n")
            log.write(f"  Fragmentations: {frags}\n")
            log.write(f"  Fragmented genera: {frag_genera}\n")
            log.write(f"  Fusions: {fus}\n")
            log.write(f"  Fused genera: {fus_genera}\n\n")

            writer.writerow([t, frags, fus])
            frag_str = "|".join(frag_genera) if frag_genera else ""
            fus_str  = "|".join(fus_genera)  if fus_genera else ""
            det_writer.writerow([t, frag_str, fus_str])

            thresholds.append(t)
            frag_list.append(frags)
            fus_list.append(fus)

            fusion_ok = check_fusion_limit(genus_to_clusters, fusion_limit)
            fragment_ok = check_fragment_limit(genus_to_clusters, fragment_limit)
            limit_status[t] = (fusion_ok, fragment_ok)

    if fusion_limit or fragment_limit:
        chosen_t, exact_match = find_best_t_by_genera(thresholds, limit_status)

        print("\nGenus-based limits detected.")
        print(f"Fusion limit genera: {fusion_limit}")
        print(f"Fragmentation limit genera: {fragment_limit}")
        print(f"Best threshold found: {chosen_t} (exact_match={exact_match})")

        best_output = os.path.join(args.outdir, "best_threshold_clusters.tsv")
        run_treecluster(args.tree, chosen_t, output_file=best_output)

        best_path = os.path.join(args.outdir, "best_threshold.txt")
        with open(best_path, "w") as bf:
            bf.write("Genus-based threshold selection\n")
            bf.write(f"Fusion limit: {fusion_limit}\n")
            bf.write(f"Fragmentation limit: {fragment_limit}\n")
            bf.write(f"Best threshold: {chosen_t}\n")
            bf.write(f"Exact match: {exact_match}\n")

    else:
        mixing_scores = {}

        for t in thresholds:
            clusters = all_clusters[t]
            mixing_scores[t] = compute_mixing_score_from_clusters(
                clusters, tree_to_species, species_to_genus
            )

        chosen_t = min(
            thresholds,
            key=lambda t: (mixing_scores[t], frag_list[thresholds.index(t)], -t)
        )

        print("\nNo genus limits provided.")
        print(f"Best threshold minimizing genus mixing: {chosen_t} "
              f"(mixing={mixing_scores[chosen_t]}, fragmentations={frag_list[thresholds.index(chosen_t)]})")

        best_output = os.path.join(args.outdir, "best_threshold_clusters.tsv")
        run_treecluster(args.tree, chosen_t, output_file=best_output)

        best_path = os.path.join(args.outdir, "best_threshold.txt")
        with open(best_path, "w") as bf:
            bf.write("Genus-mixing threshold selection\n")
            bf.write(f"Best threshold: {chosen_t}\n")
            bf.write(f"Genus mixing score: {mixing_scores[chosen_t]}\n")
            bf.write(f"Fragmentations at best t: {frag_list[thresholds.index(chosen_t)]}\n")

    # -----------------------------------------------------
    # Build GTDB matrix
    # -----------------------------------------------------
    gtdb_matrix = build_gtdb_matrix(
        all_clusters, thresholds, tree_to_species, species_to_genus, all_genera
    )

    cmap = ListedColormap(["#4daf4a", "#e41a1c", "#377eb8", "#984ea3"])

    plt.figure(figsize=(18, 8))
    img = plt.imshow(
        gtdb_matrix,
        aspect="auto",
        cmap=cmap,
        vmin=0,
        vmax=3,
        interpolation="nearest",
        origin="lower"
    )

    # Draw vertical line for chosen threshold
    if chosen_t is not None:
        plt.axhline(
            y=thresholds.index(chosen_t),
            color="black",
            linestyle="--",
            linewidth=1.5,
            label=f"Best t = {chosen_t}"
        )

    # -----------------------------------------------------
    # Legend with explanations (English)
    # -----------------------------------------------------

    best_t_line = Line2D(
        [0], [0],
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"Best threshold = {chosen_t}"
    )
    
    legend_patches = [
        mpatches.Patch(color="#4daf4a", label="OK — genus is clean (one cluster, no mixing)"),
        mpatches.Patch(color="#e41a1c", label="Fragmented — genus appears in multiple clusters"),
        mpatches.Patch(color="#377eb8", label="Fused — cluster contains multiple genera"),
        mpatches.Patch(color="#984ea3", label="Frag+Fused — genus is both fragmented and fused"),
        best_t_line
    ]

    plt.legend(
        handles=legend_patches,
        title="GTDB Category Meaning",
        loc="upper right",
        fontsize=8,
        title_fontsize=9,
        frameon=True
    )

    # -----------------------------------------------------
    # Axes and grid
    # -----------------------------------------------------
    plt.xticks(range(len(all_genera)), all_genera, rotation=45, fontsize=8)

    for x in range(len(all_genera)):
        plt.axvline(x - 0.5, color="black", linewidth=0.3, alpha=0.4)

    step = max(1, len(thresholds) // 20)
    yticks = list(range(0, len(thresholds), step))
    plt.yticks(yticks, [thresholds[i] for i in yticks], fontsize=8)

    plt.xlabel("Genus")
    plt.ylabel("Threshold")
    plt.title("GTDB-style Genus Stability Across Thresholds")
    plt.tight_layout()
    plt.savefig(gtdb_plot_path, dpi=300)
    plt.close()


    # -----------------------------------------------------
    # Classic plot
    # -----------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, frag_list, label="Fragmentations", marker="o", color="#e41a1c")
    plt.plot(thresholds, fus_list, label="Fusions", marker="o", color="#377eb8")

    if chosen_t is not None:
        plt.axvline(
            chosen_t,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label=f"Best t = {chosen_t}"
        )

    plt.xlabel("Threshold (t)")
    plt.ylabel("Count")
    plt.title("Fragmentations and Fusions Across Thresholds")
    plt.grid(True)
    plt.legend()
    plt.savefig(classic_plot_path)
    plt.close()

    # -----------------------------------------------------
    # Run AAI validation if provided
    # -----------------------------------------------------
    if args.aai:
        validate_t_with_AAI(best_output, args.aai, args.outdir)

    print(f"\nAnalysis completed. Results saved in: {args.outdir}")


# ---------------------------------------------------------
# 12. ENTRY POINT
# ---------------------------------------------------------
if __name__ == "__main__":
    main()
