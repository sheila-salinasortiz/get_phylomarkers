#!/usr/bin/env python3
"""
Compute the vertical cut height h ∈ [0,1] that reproduces exactly the
clusters defined in a TSV file, and visualize the tree with:

- Leaves colored by cluster
- Singletons (-1) in black
- A vertical cut line at height h

This extended version generates THREE figures:
1) Tree with leaf IDs
2) Tree with real species names
3) Tree with bootstrap values extracted from IQ-TREE format (/SH/UFBoot)

This script performs NO phylogenetic inference.
It only uses the tree's branch lengths to find the cut that yields the TSV partition.
"""

import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
from Bio import Phylo
import matplotlib.colors as mcolors
import re
import os


# ------------------------------------------------------------
# Extract bootstrap from IQ-TREE format: /SH/UFBoot
# ------------------------------------------------------------
def extract_bootstrap(clade):
    """
    IQ-TREE often encodes support as: /SH-aLRT/UFBoot
    Example: /1/100  → bootstrap = 100

    Biopython does NOT parse this automatically.
    We extract the last number from clade.name if present.
    """
    if clade.name is None:
        return None

    # Look for patterns like /X/Y
    m = re.findall(r"/(\d+)", clade.name)
    if m:
        return int(m[-1])  # last number = UFBoot
    return None


# ------------------------------------------------------------
# Load real species names from TXT file
# ------------------------------------------------------------
def load_bacteria_from_txt(filename):
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


# ------------------------------------------------------------
# Read TSV clusters
# ------------------------------------------------------------
def read_clusters(tsv_path):
    leaf_to_cluster = {}

    with open(tsv_path) as f:
        header = f.readline().strip().split("\t")
        i_name = header.index("SequenceName")
        i_cl = header.index("ClusterNumber")

        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split("\t")
            leaf = parts[i_name]
            cl = int(parts[i_cl])
            leaf_to_cluster[leaf] = cl

    return leaf_to_cluster


# ------------------------------------------------------------
# Compute depth of each node from the root
# ------------------------------------------------------------
def compute_depths(tree):
    depths = {}

    def walk(clade, depth):
        depths[clade] = depth
        for child in clade.clades:
            bl = child.branch_length if child.branch_length else 0.0
            walk(child, depth + bl)

    walk(tree.root, 0.0)
    return depths


# ------------------------------------------------------------
# Given a cut height h, compute the induced partition of leaves
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# Check if the cut-induced partition matches the TSV partition
# ------------------------------------------------------------
def matches_tsv(leaf_to_group, leaf_to_cluster):
    cluster_to_group = {}
    for leaf, cl in leaf_to_cluster.items():
        g = leaf_to_group[leaf]
        if cl > 0:
            if cl not in cluster_to_group:
                cluster_to_group[cl] = g
            else:
                if cluster_to_group[cl] != g:
                    return False

    group_counts = defaultdict(int)
    for leaf, g in leaf_to_group.items():
        group_counts[g] += 1

    for leaf, cl in leaf_to_cluster.items():
        if cl == -1:
            g = leaf_to_group[leaf]
            if group_counts[g] != 1:
                return False

    return True


# ------------------------------------------------------------
# Search for a cut height h ∈ [0,1]
# ------------------------------------------------------------
def find_cut_height(tree, leaf_to_cluster):
    depths = compute_depths(tree)
    unique_depths = sorted(set(depths.values()))
    candidates = []

    for d1, d2 in zip(unique_depths[:-1], unique_depths[1:]):
        h = (d1 + d2) / 2.0
        if 0.0 <= h <= 1.0:
            candidates.append(h)

    for h in candidates:
        leaf_to_group = clusters_from_cut(tree, depths, h)
        if matches_tsv(leaf_to_group, leaf_to_cluster):
            return h

    return None


# ------------------------------------------------------------
# Draw tree with original leaf names
# ------------------------------------------------------------
def draw_tree(tree, h, leaf_to_cluster, out_path):
    depths = compute_depths(tree)

    leaves = tree.get_terminals()
    leaf_positions = {leaf: i for i, leaf in enumerate(leaves)}
    max_y = len(leaves) - 1

    node_y = {}

    def compute_y(clade):
        if clade in leaf_positions:
            node_y[clade] = leaf_positions[clade]
        else:
            child_ys = []
            for child in clade.clades:
                compute_y(child)
                child_ys.append(node_y[child])
            node_y[clade] = sum(child_ys) / len(child_ys)

    compute_y(tree.root)

    fig, ax = plt.subplots(figsize=(12, 16))

    for clade in tree.find_clades(order="level"):
        x1 = depths[clade]
        y1 = node_y[clade]
        for child in clade.clades:
            x2 = depths[child]
            y2 = node_y[child]
            ax.plot([x1, x1], [y1, y2], color="black", linewidth=1)
            ax.plot([x1, x2], [y2, y2], color="black", linewidth=1)

    palette = list(mcolors.TABLEAU_COLORS.values())
    cluster_ids = sorted({c for c in leaf_to_cluster.values() if c != -1})
    cmap = {cid: palette[i % len(palette)] for i, cid in enumerate(cluster_ids)}

    label_offset = 0.02
    for leaf in leaves:
        cid = leaf_to_cluster.get(leaf.name, -1)
        color = cmap.get(cid, "black")

        x = depths[leaf]
        y = node_y[leaf]

        ax.scatter([x], [y], color=color, s=40, zorder=3)
        ax.text(x + label_offset, y, leaf.name, color=color, fontsize=10, va="center")

    if h is not None:
        ax.vlines(h, 0, max_y, colors="red", linestyles="--", linewidth=2)
        ax.text(h, max_y+0.1, f"h = {h:.4f}", color="red", ha="center", va="bottom")

    ax.set_xlabel("Branch length")
    ax.set_ylabel("Taxa index")
    ax.set_ylim(-1, max_y + 1)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------
# Draw tree with REAL species names
# ------------------------------------------------------------
def draw_tree_with_real_labels(tree, h, leaf_to_cluster, leaf_to_realname, out_path):
    depths = compute_depths(tree)

    leaves = tree.get_terminals()
    leaf_positions = {leaf: i for i, leaf in enumerate(leaves)}
    max_y = len(leaves) - 1

    node_y = {}

    def compute_y(clade):
        if clade in leaf_positions:
            node_y[clade] = leaf_positions[clade]
        else:
            child_ys = []
            for child in clade.clades:
                compute_y(child)
                child_ys.append(node_y[child])
            node_y[clade] = sum(child_ys) / len(child_ys)

    compute_y(tree.root)

    fig, ax = plt.subplots(figsize=(12, 16))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for clade in tree.find_clades(order="level"):
        x1 = depths[clade]
        y1 = node_y[clade]
        for child in clade.clades:
            x2 = depths[child]
            y2 = node_y[child]
            ax.plot([x1, x1], [y1, y2], color="black", linewidth=1)
            ax.plot([x1, x2], [y2, y2], color="black", linewidth=1)

    palette = list(mcolors.TABLEAU_COLORS.values())
    cluster_ids = sorted({c for c in leaf_to_cluster.values() if c != -1})
    cmap = {cid: palette[i % len(palette)] for i, cid in enumerate(cluster_ids)}

    label_offset = 0.025
    for leaf in leaves:
        cid = leaf_to_cluster.get(leaf.name, -1)
        color = cmap.get(cid, "black")

        x = depths[leaf]
        y = node_y[leaf]

        ax.scatter([x], [y], color=color, s=40, zorder=3)
        ax.text(
            x + label_offset,
            y,
            leaf_to_realname[leaf.name],
            color=color,
            fontsize=10,
            va="center"
        )

    if h is not None:
        ax.vlines(h, 0, max_y, colors="red", linestyles="--", linewidth=2)
        ax.text(h, max_y+0.1, f"h = {h:.4f}", color="red", ha="center", va="bottom")

    ax.set_xlabel("Branch length")
    ax.set_ylabel("Taxa index")
    ax.set_ylim(-1, max_y + 1)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------
# Draw tree WITH BOOTSTRAP VALUES
# ------------------------------------------------------------
def draw_tree_with_bootstrap(tree, h, leaf_to_cluster, out_path):
    depths = compute_depths(tree)

    leaves = tree.get_terminals()
    leaf_positions = {leaf: i for i, leaf in enumerate(leaves)}
    max_y = len(leaves) - 1

    node_y = {}

    def compute_y(clade):
        if clade in leaf_positions:
            node_y[clade] = leaf_positions[clade]
        else:
            child_ys = []
            for child in clade.clades:
                compute_y(child)
                child_ys.append(node_y[child])
            node_y[clade] = sum(child_ys) / len(child_ys)

    compute_y(tree.root)

    fig, ax = plt.subplots(figsize=(12, 16))

    for clade in tree.find_clades(order="level"):
        x1 = depths[clade]
        y1 = node_y[clade]
        for child in clade.clades:
            x2 = depths[child]
            y2 = node_y[child]

            ax.plot([x1, x1], [y1, y2], color="black", linewidth=1)
            ax.plot([x1, x2], [y2, y2], color="black", linewidth=1)

            # Extract bootstrap from IQ-TREE format
            bs = extract_bootstrap(child)
            if bs is not None:
                bx = (x1 + x2) / 2
                by = y2
                ax.text(
                    bx, by + 0.2,
                    f"{bs}",
                    fontsize=8,
                    color="gray",
                    ha="center"
                )

    palette = list(mcolors.TABLEAU_COLORS.values())
    cluster_ids = sorted({c for c in leaf_to_cluster.values() if c != -1})
    cmap = {cid: palette[i % len(palette)] for i, cid in enumerate(cluster_ids)}

    label_offset = 0.02
    for leaf in leaves:
        cid = leaf_to_cluster.get(leaf.name, -1)
        color = cmap.get(cid, "black")

        x = depths[leaf]
        y = node_y[leaf]

        ax.scatter([x], [y], color=color, s=40, zorder=3)
        ax.text(x + label_offset, y, leaf.name, color=color, fontsize=10, va="center")

    if h is not None:
        ax.vlines(h, 0, max_y, colors="red", linestyles="--", linewidth=2)
        ax.text(h, max_y+0.1, f"h = {h:.4f}", color="red", ha="center", va="bottom")

    ax.set_xlabel("Branch length")
    ax.set_ylabel("Taxa index")
    ax.set_ylim(-1, max_y + 1)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Tree cut visualization based on TSV clusters")
    parser.add_argument("--tree", required=True, help="Tree in newick format")
    parser.add_argument("--clusters", required=True, help="Clusters generated by TreeCluster.")
    parser.add_argument("--tree_labels", required=True,
                        help="TXT file with species names to use as labels")
    parser.add_argument("--out", default="tree_with_cut.pdf", help="Output pdf file")
    parser.add_argument("--outdir", default="plot_clusters", help="Output directory for PDFs")
    args = parser.parse_args()

    tree = Phylo.read(args.tree, "newick")
    leaf_to_cluster = read_clusters(args.clusters)

    real_names = load_bacteria_from_txt(args.tree_labels)

    leaf_to_realname = {}
    for leaf in tree.get_terminals():
        try:
            idx = int(leaf.name)
        except ValueError:
            raise ValueError(f"Leaf name '{leaf.name}' is not a valid integer ID")

        if not (1 <= idx <= len(real_names)):
            raise ValueError(f"Leaf ID {idx} out of range in tree_labels.list")

        leaf_to_realname[leaf.name] = real_names[idx - 1]

    h = find_cut_height(tree, leaf_to_cluster)
    print(f"Cut height found: {h}")

    # Output directory
    os.makedirs(args.outdir, exist_ok=True)

    out_path = os.path.join(args.outdir, args.out)
    draw_tree(tree, h, leaf_to_cluster, out_path)
    print(f"Tree saved to {out_path}")

    out2 = os.path.join(args.outdir, args.out.replace(".pdf", "_realnames.pdf"))
    draw_tree_with_real_labels(tree, h, leaf_to_cluster, leaf_to_realname, out2)
    print(f"Tree with real labels saved to {out2}")

    out3 = os.path.join(args.outdir, args.out.replace(".pdf", "_bootstrap.pdf"))
    draw_tree_with_bootstrap(tree, h, leaf_to_cluster, out3)
    print(f"Tree with bootstrap saved to {out3}")


if __name__ == "__main__":
    main()

