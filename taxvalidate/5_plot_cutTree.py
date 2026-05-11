#!/usr/bin/env python3
"""
Compute the vertical cut height h ∈ [0,1] that reproduces exactly the
clusters defined in a TSV file, and visualize the tree with:

- Leaves colored by cluster
- Singletons (-1) in black
- A vertical cut line at height h
- NO branch labels (bootstrap values hidden)

Additionally, this extended version generates a SECOND tree figure where
leaf labels are replaced by real species names extracted from a TXT file
(tree_labels.list), using a strict normalization rule.

This script performs NO phylogenetic inference.
It only uses the tree's branch lengths to find the cut that yields the TSV partition.
"""

import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
from Bio import Phylo
import matplotlib.colors as mcolors
import re


# ------------------------------------------------------------
# Load real species names from TXT file
# ------------------------------------------------------------
def load_bacteria_from_txt(filename):
    """
    Extracts normalized species names from a TXT file.

    Rule: ALWAYS take the first 2 words inside brackets,
    replacing '_' with a space.

    Examples:
      [Antrihabitans_sp._YC3-6] → Antrihabitans sp.
      [Jongsikchunia_kroppenstedtii_DSM_45133] → Jongsikchunia kroppenstedtii
      [Mycobacterium_avium_subsp._avium] → Mycobacterium avium

    If a line does not contain brackets, the whole line is used as-is.
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


# ------------------------------------------------------------
# Read TSV clusters
# ------------------------------------------------------------
def read_clusters(tsv_path):
    """
    Reads the TSV file produced by TreeCluster and returns:

      leaf_to_cluster: dict {leaf_name -> cluster_id}

    Cluster -1 means singleton.
    """
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
    """
    Computes the distance from the root to each clade.

    Returns:
      depths: dict {clade -> depth_from_root}
    """
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
    """
    A vertical cut at height h splits the tree into components.

    Returns:
      leaf_to_group: dict {leaf_name -> group_id}
    """
    cut_roots = set()

    # Identify nodes where the cut intersects a branch
    def mark(clade):
        for child in clade.clades:
            if depths[clade] < h <= depths[child]:
                cut_roots.add(child)
            mark(child)

    mark(tree.root)

    # If no branch is cut, the whole tree is one group
    if not cut_roots:
        cut_roots = {tree.root}

    leaf_to_group = {}
    group_id = 0

    # Assign group IDs to leaves under each cut root
    for root in cut_roots:
        group_id += 1
        for leaf in root.get_terminals():
            leaf_to_group[leaf.name] = group_id

    # Any leaf not assigned (cut above the root) becomes its own group
    for leaf in tree.get_terminals():
        if leaf.name not in leaf_to_group:
            group_id += 1
            leaf_to_group[leaf.name] = group_id

    return leaf_to_group


# ------------------------------------------------------------
# Check if the cut-induced partition matches the TSV partition
# ------------------------------------------------------------
def matches_tsv(leaf_to_group, leaf_to_cluster):
    """
    Conditions:
    - All leaves with same cluster > 0 must share the same group
    - Leaves with cluster = -1 must be alone in their group
    """
    # Check consistency for clusters > 0
    cluster_to_group = {}
    for leaf, cl in leaf_to_cluster.items():
        g = leaf_to_group[leaf]
        if cl > 0:
            if cl not in cluster_to_group:
                cluster_to_group[cl] = g
            else:
                if cluster_to_group[cl] != g:
                    return False

    # Check singletons
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
# Search for a cut height h ∈ [0,1] that reproduces the TSV partition
# ------------------------------------------------------------
def find_cut_height(tree, leaf_to_cluster):
    """
    Searches all candidate heights between unique node depths and
    returns the first height that reproduces the TSV partition.
    """
    depths = compute_depths(tree)

    # Candidate cut heights = midpoints between unique node depths
    unique_depths = sorted(set(depths.values()))
    candidates = []
    for d1, d2 in zip(unique_depths[:-1], unique_depths[1:]):
        h = (d1 + d2) / 2.0
        if 0.0 <= h <= 1.0:
            candidates.append(h)

    # Test each candidate
    for h in candidates:
        leaf_to_group = clusters_from_cut(tree, depths, h)
        if matches_tsv(leaf_to_group, leaf_to_cluster):
            return h

    return None


# ------------------------------------------------------------
# Draw tree with original leaf names
# ------------------------------------------------------------
def draw_tree(tree, h, leaf_to_cluster, out_path):
    """
    Fully manual tree drawing:
    - Computes x = depth from root
    - Computes y = leaf index
    - Draws branches manually
    - Colors leaf names by cluster
    - Hides bootstrap labels
    - Draws vertical cut
    """
    depths = compute_depths(tree)

    # Assign y positions to leaves (even spacing)
    leaves = tree.get_terminals()
    leaf_positions = {leaf: i for i, leaf in enumerate(leaves)}
    max_y = len(leaves) - 1

    # Compute y for internal nodes = average of children
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

    # Prepare figure
    fig, ax = plt.subplots(figsize=(12, 16))

    # Draw branches
    for clade in tree.find_clades(order="level"):
        x1 = depths[clade]
        y1 = node_y[clade]
        for child in clade.clades:
            x2 = depths[child]
            y2 = node_y[child]
            ax.plot([x1, x1], [y1, y2], color="black", linewidth=1)
            ax.plot([x1, x2], [y2, y2], color="black", linewidth=1)

    # Color map for clusters
    palette = list(mcolors.TABLEAU_COLORS.values())
    cluster_ids = sorted({c for c in leaf_to_cluster.values() if c != -1})
    cmap = {cid: palette[i % len(palette)] for i, cid in enumerate(cluster_ids)}

    # Draw colored leaf names
    label_offset = 0.02
    for leaf in leaves:
        cid = leaf_to_cluster.get(leaf.name, -1)
        color = cmap.get(cid, "black")

        x = depths[leaf]
        y = node_y[leaf]

        ax.scatter([x], [y], color=color, s=40, zorder=3)
        ax.text(x + label_offset, y, leaf.name, color=color, fontsize=10, va="center")

    # Draw vertical cut
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
    """
    Same as draw_tree(), but leaf labels are replaced by real species names.
    This is useful for publication-quality figures.
    """
    depths = compute_depths(tree)

    leaves = tree.get_terminals()
    leaf_positions = {leaf: i for i, leaf in enumerate(leaves)}
    max_y = len(leaves) - 1

    # Compute y for internal nodes
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

    # Remove top and right spines for a cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


    # Draw branches
    for clade in tree.find_clades(order="level"):
        x1 = depths[clade]
        y1 = node_y[clade]
        for child in clade.clades:
            x2 = depths[child]
            y2 = node_y[child]
            ax.plot([x1, x1], [y1, y2], color="black", linewidth=1)
            ax.plot([x1, x2], [y2, y2], color="black", linewidth=1)

    # Colors
    palette = list(mcolors.TABLEAU_COLORS.values())
    cluster_ids = sorted({c for c in leaf_to_cluster.values() if c != -1})
    cmap = {cid: palette[i % len(palette)] for i, cid in enumerate(cluster_ids)}

    # Draw real labels
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

    # Vertical cut
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
    args = parser.parse_args()

    # Load tree and clusters
    tree = Phylo.read(args.tree, "newick")
    leaf_to_cluster = read_clusters(args.clusters)

    # Load real species names
    real_names = load_bacteria_from_txt(args.tree_labels)

    # Map leaf ID → real name (correct and order-independent)
    leaf_to_realname = {}
    for leaf in tree.get_terminals():
        try:
            idx = int(leaf.name)
        except ValueError:
            raise ValueError(f"Leaf name '{leaf.name}' is not a valid integer ID")

        if not (1 <= idx <= len(real_names)):
            raise ValueError(f"Leaf ID {idx} out of range in tree_labels.list")

        leaf_to_realname[leaf.name] = real_names[idx - 1]

    # Compute cut height
    h = find_cut_height(tree, leaf_to_cluster)
    print(f"Cut height found: {h}")

    # Draw tree with original labels
    draw_tree(tree, h, leaf_to_cluster, args.out)
    print(f"Tree saved to {args.out}")

    # Draw tree with real species names
    out2 = args.out.replace(".pdf", "_realnames.pdf")
    draw_tree_with_real_labels(tree, h, leaf_to_cluster, leaf_to_realname, out2)
    print(f"Tree with real labels saved to {out2}")


if __name__ == "__main__":
    main()
