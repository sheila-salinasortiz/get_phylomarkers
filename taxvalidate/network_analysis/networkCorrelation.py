#!/usr/bin/env python3
import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import pearsonr
import networkx as nx
import igraph as ig
from matplotlib.lines import Line2D

# ---------------------------------------------------------
# LOAD AAI / AF MATRIX
# ---------------------------------------------------------
def load_matrix(path: str) -> pd.DataFrame:
    """
    Load a similarity matrix (AAI/AF/POCP) from a text/TSV file.
    - Drops non-numeric first column if present.
    - Converts all values to numeric.
    - Assigns row/column labels as '1'..'N'.
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
# LOAD ML DISTANCE MATRIX AND CONVERT TO SIMILARITY
# ---------------------------------------------------------
def load_mldist_as_similarity(path: str) -> pd.DataFrame:
    """
    Load a PHYLIP-style ML distance matrix and convert it to similarity:
    similarity = 1 / (1 + distance)
    This transformation is required to keep the biological order for
    the Pearson correlation calculus.
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
# LOAD CLUSTERS
# ---------------------------------------------------------
def load_clusters(path: str) -> dict:
    """
    Load cluster assignments from TSV.
    Returns dict: { genome_id : cluster_id }
    """
    df = pd.read_csv(path, sep="\t")
    df["SequenceName"] = df["SequenceName"].astype(str)
    return dict(zip(df["SequenceName"], df["ClusterNumber"]))

# ---------------------------------------------------------
# LOAD TAXONOMY
# ---------------------------------------------------------
def load_taxonomy(path: str) -> list:
    """
    Load genus names from CSV.
    Returns list indexed by genome ID (1..N).
    """
    df = pd.read_csv(path)
    return df["GENUS"].astype(str).tolist()

# ---------------------------------------------------------
# PER-GENOME MEAN MATRIX WITHIN CLUSTER
# ---------------------------------------------------------
def internal_mean_matrix_clustered(matrix: pd.DataFrame, clusters: dict) -> list:
    """
    Compute mean MATRIX of each genome to other members of its cluster.
    - Simple, stable, biologically interpretable.
    """
    genomes = matrix.index.tolist()
    mean_values = []

    for g in genomes:
        cid = clusters[g]

        if cid == -1:
            mean_values.append(np.nan)
            continue

        members = [x for x, c in clusters.items() if c == cid and x != g]

        vals = matrix.loc[g, members].values
        vals = vals[~np.isnan(vals)]

        mean_values.append(np.mean(vals) if len(vals) else np.nan)

    return mean_values

# ---------------------------------------------------------
# GRAPHIA-STYLE GLOBAL CORRELATION
# ---------------------------------------------------------
def graphia_style_correlation(matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Compute global Pearson correlation exactly like Graphia:
    - Uses full-length vectors (all genomes).
    - Removes diagonal.
    - Pearson correlation with NaN masking.
    - Applies positive polarity (negatives → 0).
    """
    genomes = matrix.index.tolist()

    # Remove diagonal (AAI/AF/POCP self = 100)
    np.fill_diagonal(matrix.values, np.nan)

    corr_mat = pd.DataFrame(index=genomes, columns=genomes, dtype=float)

    for i in genomes:
        vec_i = matrix.loc[i, :].values

        for j in genomes:
            if i == j:
                corr_mat.loc[i, j] = 1.0
                continue

            vec_j = matrix.loc[j, :].values

            mask = ~np.isnan(vec_i) & ~np.isnan(vec_j)
            if mask.sum() < 3:
                corr_mat.loc[i, j] = np.nan
                continue

            corr = pearsonr(vec_i[mask], vec_j[mask])[0]

            # Positive polarity
            if corr < 0:
                corr = 0.0

            corr_mat.loc[i, j] = corr

    return corr_mat

# ---------------------------------------------------------
# CLUSTER CORR MATRICES DERIVED FROM GLOBAL CORR
# ---------------------------------------------------------
def cluster_corrs_from_global(corr_global: pd.DataFrame, clusters: dict, 
                              OUTPUT_DIR: str):
    """
    Derive per-cluster correlation matrices by subsetting the global
    correlation matrix. No new Pearson is computed.
    - Values are EXACTLY the same as in correlation_global.tsv.
    - Only intra-cluster pairs are kept.
    - Each cluster matrix is saved to TSV.
    """
    genomes = corr_global.index.tolist()
    cluster_ids = sorted(set(clusters.values()))
    cluster_corrs = {}

    for cid in cluster_ids:
        if cid == -1:
            cluster_corrs[cid] = pd.DataFrame()
            continue

        members = [g for g in genomes if clusters[g] == cid]

        if len(members) < 2:
            cluster_corrs[cid] = pd.DataFrame()
            continue

        sub = corr_global.loc[members, members].copy()
        sub.to_csv(os.path.join(OUTPUT_DIR, f"cluster_{cid}_correlation_from_global.tsv"), sep="\t")
        cluster_corrs[cid] = sub

    return cluster_corrs

# ---------------------------------------------------------
# PER-GENOME MEAN CORRELATION WITHIN CLUSTER (FROM GLOBAL)
# ---------------------------------------------------------
def per_genome_mean_corr_in_cluster(cluster_corrs, clusters):
    """
    For each genome, compute the mean Pearson correlation
    to other members of its own cluster (using cluster_corrs
    derived from the global correlation matrix).
    """
    genomes = sorted(clusters.keys(), key=lambda g: int(g))
    values = []

    for g in genomes:
        cid = clusters[g]
        corr_mat = cluster_corrs.get(cid, pd.DataFrame())

        if corr_mat.empty or g not in corr_mat.index:
            values.append(np.nan)
            continue

        row = corr_mat.loc[g, :].drop(labels=[g], errors="ignore")
        row = row[~np.isnan(row)]

        values.append(row.mean() if len(row) else np.nan)

    return pd.DataFrame({"mean_corr": values}, index=genomes)

# ---------------------------------------------------------
# COLOR MAP FOR GENERA
# ---------------------------------------------------------
def get_genus_color_map(genus_list):
    """
    Assign a unique color to each genus using a color-blind safe palette.
    """
    unique_genera = sorted(set(genus_list))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_genera)))
    return dict(zip(unique_genera, colors))

# ---------------------------------------------------------
# SCATTER PLOT OF MEAN MATRIX
# ---------------------------------------------------------
def plot_mean_matrix(values, clusters, genus_list, output_pdf, matrix_name):
    """
    Scatter plot of per-genome mean AAI/AF/ML within cluster.
    Genomes ordered by cluster ID.
    """
    genomes = list(values.index)
    ordered = sorted(genomes, key=lambda g: clusters[g])
    values = values.loc[ordered]
    genus_ordered = [genus_list[int(g) - 1] for g in ordered]

    unique_genera = sorted(set(genus_ordered))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_genera)))
    genus_to_color = dict(zip(unique_genera, colors))
    point_colors = [genus_to_color[g] for g in genus_ordered]

    # Dynamic labels
    y_label = f"Mean {matrix_name}"
    title = f"Per-genome mean {matrix_name} within cluster"

    with PdfPages(output_pdf) as pdf:
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Genome index (ordered by cluster)")
        ax.set_ylabel(y_label)

        x = np.arange(len(ordered))
        ax.scatter(x, values[f"mean_{matrix_name}"], c=point_colors, s=40)

        ax.set_xticks(x)
        ax.set_xticklabels(ordered, rotation=90, fontsize=6)
        ax.grid(True, alpha=0.3)

        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w',
                       label=gen, markerfacecolor=genus_to_color[gen], markersize=8)
            for gen in unique_genera
        ]

        ax.legend(handles=legend_elements,
                  title="Genus",
                  loc="upper left",
                  bbox_to_anchor=(1.02, 1),
                  fontsize=8,
                  title_fontsize=10)

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

# ---------------------------------------------------------
# SCATTER PLOT OF MEAN CORRELATION WITHIN CLUSTER
# ---------------------------------------------------------
def plot_mean_corr(values, clusters, genus_list, output_pdf):
    """
    Scatter plot of per-genome mean Pearson correlation within cluster
    (using global correlation restricted to cluster members).
    """
    genomes = list(values.index)
    ordered = sorted(genomes, key=lambda g: clusters[g])
    values = values.loc[ordered]
    genus_ordered = [genus_list[int(g) - 1] for g in ordered]

    unique_genera = sorted(set(genus_ordered))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_genera)))
    genus_to_color = dict(zip(unique_genera, colors))
    point_colors = [genus_to_color[g] for g in genus_ordered]

    with PdfPages(output_pdf) as pdf:
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.set_title("Per-genome mean Pearson correlation within cluster", fontsize=14)
        ax.set_xlabel("Genome index (ordered by cluster)")
        ax.set_ylabel("Mean Pearson correlation")

        x = np.arange(len(ordered))
        ax.scatter(x, values["mean_corr"], c=point_colors, s=40)

        ax.set_xticks(x)
        ax.set_xticklabels(ordered, rotation=90, fontsize=6)
        ax.grid(True, alpha=0.3)

        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w',
                       label=gen, markerfacecolor=genus_to_color[gen], markersize=8)
            for gen in unique_genera
        ]

        ax.legend(handles=legend_elements,
                  title="Genus",
                  loc="upper left",
                  bbox_to_anchor=(1.02, 1),
                  fontsize=8,
                  title_fontsize=10)

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

# ---------------------------------------------------------
# 2D CLUSTER-ISLAND NETWORK USING GLOBAL CORR
# ---------------------------------------------------------
def plot_cluster_islands_from_global(cluster_corrs, clusters, genus_list,
                                     OUTPUT_DIR, output_pdf, percentile=60):
    """
    Create a 2D network where:
    - Edge weights come from the GLOBAL correlation matrix (Graphia-style).
    - Only intra-cluster edges are drawn.
    - Each cluster has its own FR layout (islands).
    - Threshold is percentile-based per cluster (on global corr values).
    This preserves:
    - The same correlation values as the global matrix.
    - The visual separation of clusters as islands.
    """

    total_genomes = len([g for g in clusters if clusters[g] != -1])
    total_clusters = len([cid for cid, mat in cluster_corrs.items() if not mat.empty])

    NODE_SIZE = max(40, 140 - total_genomes * 0.25)
    LABEL_SIZE = max(6, 14 - total_genomes * 0.015)
    EDGE_WIDTH = max(0.3, 1.2 - total_genomes * 0.004)

    CLUSTER_SPACING = 7 + np.log10(max(3, total_clusters)) * 8

    def scale_for_cluster(n):
        return 2.2 + np.log10(max(3, n)) * 3.0

    fig, ax = plt.subplots(figsize=(22, 14))

    genus_to_color = get_genus_color_map(genus_list)
    unique_genera = sorted(set(genus_list))

    x_offset = 0.0

    for cid, corr_mat in cluster_corrs.items():

        if corr_mat.empty or corr_mat.shape[0] < 2:
            continue

        members = corr_mat.index.tolist()
        n_members = len(members)

        vals = corr_mat.values.flatten()
        vals = vals[~np.isnan(vals)]
        vals = vals[vals < 1.0]
        if len(vals) == 0:
            continue

        threshold = np.nanpercentile(vals, percentile)

        edges = []
        for i in members:
            for j in members:
                if i < j:
                    c = corr_mat.loc[i, j]
                    if not np.isnan(c) and c >= threshold:
                        edges.append((members.index(i), members.index(j)))

        g = ig.Graph()
        g.add_vertices(n_members)
        g.add_edges(edges)

        layout = g.layout_fruchterman_reingold()

        ys_raw = np.array([layout[i][1] for i in range(n_members)])
        ys_norm = (ys_raw - ys_raw.min()) / (ys_raw.max() - ys_raw.min() + 1e-9)

        VERTICAL_SCALE = 5.0
        VERTICAL_SPACING = 7.0

        ys = ys_norm * VERTICAL_SCALE + (cid * VERTICAL_SPACING)

        SCALE_FACTOR = scale_for_cluster(n_members)
        xs = [(layout[i][0] * SCALE_FACTOR) + x_offset for i in range(n_members)]

        for (i, j) in edges:
            ax.plot([xs[i], xs[j]], [ys[i], ys[j]],
                    color="black", linewidth=EDGE_WIDTH, alpha=0.4)

        node_colors = [genus_to_color[genus_list[int(m)-1]] for m in members]
        ax.scatter(xs, ys, c=node_colors, s=NODE_SIZE,
                   edgecolors="black", linewidths=0.6)

        for idx, gname in enumerate(members):
            ax.text(xs[idx], ys[idx] + 0.1, gname,
                    fontsize=LABEL_SIZE, ha="center", va="bottom")

        x_offset += CLUSTER_SPACING

    legend_elements = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=genus_to_color[gen], markersize=10,
               label=gen)
        for gen in unique_genera
    ]

    ax.legend(handles=legend_elements,
              title="Genus (color-blind safe)",
              bbox_to_anchor=(1.02, 1),
              loc="upper left",
              fontsize=8,
              title_fontsize=10)

    ax.set_title(f"2D Correlation Network (Cluster islands, global corr, percentile {percentile})",
                 fontsize=18)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Global and cluster-based correlation analysis")
    parser.add_argument("--aai", help="AAI matrix file")
    parser.add_argument("--af", help="AF matrix file")
    parser.add_argument("--pocp", help="POCP matrix file")
    parser.add_argument("--ml", help="ML .mldist file (PHYLIP format)")
    parser.add_argument("--clusters", required=True, help="Cluster TSV file")
    parser.add_argument("--taxonomy", required=True, help="Taxonomy CSV file")
    parser.add_argument("--outdir", default="networkAnalysis", 
                        help="Output directory for all results.")
    args = parser.parse_args()

    # Create output directory
    OUTPUT_DIR = args.outdir
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Initialize logging AFTER creating the directory
    logging.basicConfig(
        filename=f"{OUTPUT_DIR}/network.log",
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    # Log full command line
    logging.info("Command used: " + " ".join(sys.argv))

    # Load similarity matrix
    if args.aai:
        matrix = load_matrix(args.aai)
        matrix_name = "AAI"
        logging.info("Using AAI matrix.")
    elif args.af:
        matrix = load_matrix(args.af)
        matrix_name = "AF"
        logging.info("Using AF matrix.")
    elif args.pocp:
        matrix = load_matrix(args.pocp)
        matrix_name = "POCP"
        logging.info("Using POCP matrix.")
    elif args.ml:
        matrix = load_mldist_as_similarity(args.ml)
        matrix_name = "ML_similarity"
        logging.info("Using ML matrix (converted to similarity).")
    else:
        raise ValueError("You must provide one matrix: --aai or --af or --ml")

    clusters = load_clusters(args.clusters)
    genus_list = load_taxonomy(args.taxonomy)

    # 1) Per-genome mean matrix within cluster
    # Compute per-genome mean similarity within cluster
    mean_values = internal_mean_matrix_clustered(matrix, clusters)

    # Create dataframe with the correct column name depending on matrix type
    mean_df = pd.DataFrame(
        {f"mean_{matrix_name}": mean_values},
        index=matrix.index
    )

    mean_df.to_csv(
        os.path.join(OUTPUT_DIR, f"mean_{matrix_name}_per_genome.tsv"),
        sep="\t"
    )

    plot_mean_matrix(
        mean_df,
        clusters,
        genus_list,
        f"{OUTPUT_DIR}/mean_{matrix_name}_plot.pdf",
        matrix_name
    )


    # 2) Global Graphia-style correlation
    corr_global = graphia_style_correlation(matrix)
    corr_global.to_csv(os.path.join(OUTPUT_DIR, "correlation_global.tsv"), sep="\t")

    # 3) Cluster correlation matrices derived from global
    cluster_corrs = cluster_corrs_from_global(corr_global, clusters, OUTPUT_DIR)

    # 4) Per-genome mean correlation within cluster (from global)
    mean_corr = per_genome_mean_corr_in_cluster(cluster_corrs, clusters)
    mean_corr.to_csv(os.path.join(OUTPUT_DIR, "per_genome_mean_corr_in_cluster.tsv"), sep="\t")

    plot_mean_corr(
        mean_corr,
        clusters,
        genus_list,
        os.path.join(OUTPUT_DIR, "mean_corr_plot.pdf")
    )

    # 5) 2D cluster-island network using global correlation values
    plot_cluster_islands_from_global(
        cluster_corrs,
        clusters,
        genus_list,
        OUTPUT_DIR,
        os.path.join(OUTPUT_DIR, "network_2D_clusters_from_global.pdf"),
        percentile=60
    )

    logging.info("Analysis completed.")

if __name__ == "__main__":
    main()
