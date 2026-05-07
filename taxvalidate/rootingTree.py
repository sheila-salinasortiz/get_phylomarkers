from Bio import Phylo
import argparse

def main():
    parser = argparse.ArgumentParser(description="Root a Newick tree using an optional outgroup.")
    parser.add_argument("newick_file", help="Input Newick tree file")
    parser.add_argument("-o", "--outgroup", help="Outgroup name (optional)")
    parser.add_argument("-out", "--output", default="rooted_tree.nwk",
                        help="Output file for the rooted tree (default: rooted_tree.nwk)")
    args = parser.parse_args()

    # Load the tree
    tree = Phylo.read(args.newick_file, "newick")

    # Rooting logic
    if args.outgroup:
        # Root tree using the specified outgroup
        # BioPython automatically finds the clade matching the name
        tree.root_with_outgroup(args.outgroup)
    else:
        # Midpoint rooting
        tree.root_at_midpoint()

    # Write the rooted tree to a Newick file
    Phylo.write(tree, args.output, "newick")
    print(f"Rooted tree saved to: {args.output}")

if __name__ == "__main__":
    main()
