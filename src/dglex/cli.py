import argparse
import sys
import os
from typing import Any, Dict


def load_config(config_path: str = "dglex.yaml") -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    """
    if os.path.exists(config_path):
        try:
            import yaml
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                if config and isinstance(config, dict) and "view" in config:
                    return config["view"]
        except Exception as e:
            print(f"Warning: Failed to load configuration from {config_path}: {e}", file=sys.stderr)
    return {}


def merge_config(args: argparse.Namespace, config: Dict[str, Any]) -> argparse.Namespace:
    """
    Merge command line arguments with configuration file settings.
    """
    import json
    args_dict = vars(args)

    settings_to_merge = {
        "index": "index",
        "title": "title",
        "edge_weight": "edge_weight",
        "node_palette": "node_palette",
        "edge_palette": "edge_palette",
        "output": "output",
        "figsize": "figsize",
        "reverse_etypes": "reverse_etypes",
    }

    defaults = {
        "index": 0,
        "title": None,
        "edge_weight": None,
        "node_palette": "tab10",
        "edge_palette": "tab10",
        "output": None,
        "figsize": None,
        "reverse_etypes": None,
    }

    for arg_name, config_key in settings_to_merge.items():
        if arg_name in args_dict and config_key in config:
            if args_dict[arg_name] == defaults.get(arg_name):
                if config_key == "figsize" and isinstance(config[config_key], list):
                    args_dict[arg_name] = tuple(config[config_key])
                else:
                    args_dict[arg_name] = config[config_key]

    if args.reverse_etypes and isinstance(args.reverse_etypes, str):
        try:
            args_dict["reverse_etypes"] = json.loads(args.reverse_etypes)
        except json.JSONDecodeError as e:
            print(f"Error parsing --reverse-etypes: {e}", file=sys.stderr)
            args_dict["reverse_etypes"] = None

    return args


def view(args):
    """
    Execute the view sub-command.
    """
    # Lazy imports for heavy libraries
    import dgl
    import matplotlib.pyplot as plt
    from dglex.visualisation.plot import plot_graph

    # Load and merge configuration
    config = load_config()
    args = merge_config(args, config)

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
    
    # Ensure figsize is a tuple if provided
    figsize_arg = args.figsize
    if figsize_arg is not None and isinstance(figsize_arg, list):
        figsize_arg = tuple(figsize_arg)
    
    if figsize_arg is None:
        figsize_arg = (6, 4)
    
    try:
        ax = plot_graph(
            g,
            title=args.title or f"Graph {args.index} from {args.path}",
            edge_weight_name=args.edge_weight,
            node_palette=args.node_palette,
            edge_palette=args.edge_palette,
            figsize=figsize_arg,
            reverse_etypes=args.reverse_etypes,
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
    view_parser.add_argument("--figsize", type=int, nargs=2, help="Figure size as width height (optional)")
    view_parser.add_argument("--reverse-etypes", type=str, help="Reverse edge types as a JSON string (e.g., '{\"click\": \"clicked-by\"}')")

    args = parser.parse_args()

    if args.command == "view":
        view(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
