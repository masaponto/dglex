import argparse
import dgl
import matplotlib.pyplot as plt
from dglex.visualisation.plot import plot_graph


def view(args):
    graphs, _ = dgl.load_graphs(args.path)
    if not graphs:
        print(f"No graphs found in {args.path}")
        return

    if args.index >= len(graphs):
        print(f"Index {args.index} is out of range. The file contains {len(graphs)} graphs.")
        return

    g = graphs[args.index]
    
    ax = plot_graph(
        g,
        title=args.title or f"Graph {args.index} from {args.path}",
        edge_weight_name=args.edge_weight,
    )

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
    view_parser.add_argument("--output", "-o", type=str, help="Path to save the plot image (optional)")

    args = parser.parse_args()

    if args.command == "view":
        view(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
