import argparse
import dgl
import matplotlib.pyplot as plt
import sys
from dglex.visualisation.plot import plot_graph


def view(args):
    try:
        graphs, _ = dgl.load_graphs(args.path)
    except Exception as e:
        print(f"Error loading graph file: {e}", file=sys.stderr)
        return

    if not graphs:
        print(f"No graphs found in {args.path}", file=sys.stderr)
        return

    if args.index >= len(graphs):
        print(f"Index {args.index} is out of range. The file contains {len(graphs)} graphs.", file=sys.stderr)
        return

    g = graphs[args.index]
    
    try:
        ax = plot_graph(
            g,
            title=args.title or f"Graph {args.index} from {args.path}",
            edge_weight_name=args.edge_weight,
            node_palette=args.node_palette,
            edge_palette=args.edge_palette,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("Hint: Please specify a valid Seaborn/Matplotlib palette name (e.g., 'viridis', 'magma', 'Set2').", file=sys.stderr)
        return

    if args.output:
        plt.savefig(args.output)
        print(f"Saved preview to {args.output}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="dglex: DGL extension library CLI tool")
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # view sub-command
    view_parser = subparsers.add_parser("view", help="View a DGL graph file")
    view_parser.add_argument("path", type=str, help="Path to the DGL graph (.bin) file")
    view_parser.add_argument("--index", "-i", type=int, default=0, help="Index of the graph to visualize (default: 0)")
    view_parser.add_argument("--title", "-t", type=str, help="Title of the plot")
    view_parser.add_argument("--edge-weight", "-w", type=str, help="Name of the edge weight data field")
    view_parser.add_argument("--node-palette", type=str, default="tab10", help="Seaborn color palette for nodes (default: tab10)")
    view_parser.add_argument("--edge-palette", type=str, default="tab10", help="Seaborn color palette for edges (default: tab10)")
    view_parser.add_argument("--output", "-o", type=str, help="Path to save the plot image (optional)")

    args = parser.parse_args()

    if args.command == "view":
        view(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
