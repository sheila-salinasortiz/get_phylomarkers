from Bio import Phylo
import argparse
import subprocess
import os

def main():
    parser = argparse.ArgumentParser(
        description="Root a Newick tree using an optional outgroup and optionally scale it with PhyloRank."
    )
    parser.add_argument("--tree", 
                        required=True, help="Input Newick tree file")
    parser.add_argument("-o", "--outgroup", help="Outgroup name (optional)")
    parser.add_argument(
        "-out", "--output", default="rooted_tree.nwk",
        help="Output file for the rooted tree (default: rooted_tree.nwk)"
    )
    parser.add_argument(
        "--scale", action="store_true",
        help="Normalize the rooted tree using 'phylorank scale_tree'"
    )
    parser.add_argument(
        "--scaled_output", default="scaled_rooted_tree.nwk",
        help="Output file for the scaled tree (default: scaled_rooted_tree.nwk)"
    )
    args = parser.parse_args()

    # Load the input Newick tree
    tree = Phylo.read(args.tree, "newick")

    # Apply rooting logic depending on user input
    if args.outgroup:
        # Root the tree using the specified outgroup
        tree.root_with_outgroup(args.outgroup)
    else:
        # Apply midpoint rooting when no outgroup is provided
        tree.root_at_midpoint()

    # Write the rooted tree to file
    Phylo.write(tree, args.output, "newick")
    print(f"Rooted tree saved to: {args.output}")

    # Optional scaling step using PhyloRank
    if args.scale:
        print("Scaling tree using PhyloRank...")

        cmd = [
            "phylorank",
            "scale_tree",
            args.output,
            args.scaled_output
        ]

        try:
            # Execute the external PhyloRank command
            subprocess.run(cmd, check=True)
            print(f"Scaled tree saved to: {args.scaled_output}")

        except FileNotFoundError:
            # PhyloRank is not installed or not available in PATH
            print("ERROR: 'phylorank' is not installed or not found in the system PATH.")

        except subprocess.CalledProcessError:
            # PhyloRank execution failed
            print("ERROR: Failed to execute 'phylorank scale_tree'.")

if __name__ == "__main__":
    main()
