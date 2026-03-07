import argparse
import os
import sys
from typing import Any, Dict, Literal

SamplingAction = Literal["continue", "random", "hub", "abort"]


def _decide_sampling_action(
    num_nodes: int,
    num_edges: int,
    limits: Dict[str, Any],
    force: bool,
    is_tty: bool,
    has_output: bool,
    user_choice: str,
) -> SamplingAction:
    """グラフサイズとユーザー選択からアクションを決定する純粋関数（I/O なし）。

    Args:
        num_nodes (int): グラフのノード数。
        num_edges (int): グラフのエッジ数。
        limits (Dict[str, Any]): max_nodes, max_edges を含む制限設定。
        force (bool): --force フラグが指定されているか。
        is_tty (bool): 標準入力が TTY か。
        has_output (bool): 出力先ファイルが指定されているか。
        user_choice (str): ユーザーの入力（"1"/"2"/"3"/"4"）。

    Returns:
        SamplingAction: "continue", "random", "hub", "abort" のいずれか。
    """
    is_large = num_nodes > limits["max_nodes"] or num_edges > limits["max_edges"]
    if not is_large or force:
        return "continue"
    if not is_tty or has_output:
        return "random"
    if user_choice == "1":
        return "continue"
    elif user_choice == "3":
        return "hub"
    elif user_choice == "4":
        return "abort"
    else:
        return "random"


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

    # Merge limits
    default_limits = {
        "max_nodes": 500,
        "max_edges": 2000,
        "sample_size": 5,
        "sample_fanouts": [10],
    }

    args_dict["limits"] = default_limits.copy()
    if "limits" in config and isinstance(config["limits"], dict):
        for key, value in config["limits"].items():
            if key in default_limits:
                args_dict["limits"][key] = value

    return args


def view(args):
    """
    Execute the view sub-command with safety checks for large graphs.
    """
    import dgl
    import matplotlib.pyplot as plt

    from dglex.visualisation.graph_ops import get_random_nodes, get_top_k_degree_nodes
    from dglex.visualisation.plot import plot_graph, plot_subgraph_with_neighbors

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

    # Check graph size
    num_nodes = g.num_nodes() if g.is_homogeneous else sum(g.num_nodes(ntype) for ntype in g.ntypes)
    num_edges = g.num_edges() if g.is_homogeneous else sum(g.num_edges(etype) for etype in g.canonical_etypes)

    limits = args.limits
    is_large = num_nodes > limits["max_nodes"] or num_edges > limits["max_edges"]

    user_choice = ""
    if is_large and not args.force:
        print("\nWarning: Large graph detected!")
        print(f"- Nodes: {num_nodes:,}")
        print(f"- Edges: {num_edges:,}")
        print("\nRendering this graph may be slow or cause memory issues.")

        if sys.stdin.isatty() and not args.output:
            print("\nHow would you like to proceed?")
            print("[1] Continue: Render the full graph anyway")
            print("[2] Random Sample (Default): Pick representative nodes randomly")
            print("[3] Hub Sample: Pick nodes with highest degrees")
            print("[4] Abort: Cancel the operation")
            user_choice = input("\nPlease select [1/2/3/4] (default 2): ").strip()
        else:
            # Non-interactive or output specified: safety first
            print("Action: Automatic random sampling applied for safety. Use --force to override.", file=sys.stderr)

    action = _decide_sampling_action(
        num_nodes, num_edges, limits, args.force,
        sys.stdin.isatty(), bool(args.output), user_choice
    )
    if action == "abort":
        print("Operation aborted.")
        return

    # Execution phase
    try:
        if action == "continue":
            # Ensure figsize is a tuple if provided
            figsize_arg = args.figsize
            if figsize_arg is not None and isinstance(figsize_arg, list):
                figsize_arg = tuple(figsize_arg)
            if figsize_arg is None:
                figsize_arg = (6, 4)

            plot_graph(
                g,
                title=args.title or f"Graph {args.index} from {args.path}",
                edge_weight_name=args.edge_weight,
                node_palette=args.node_palette,
                edge_palette=args.edge_palette,
                figsize=figsize_arg,
                reverse_etypes=args.reverse_etypes,
            )
        else:
            # Sampling path
            sample_size = limits["sample_size"]
            sample_fanouts = limits["sample_fanouts"]

            if action == "random":
                target_nodes = get_random_nodes(g, k=sample_size)
                mode_str = "Randomly selected"
            else: # hub
                target_nodes = get_top_k_degree_nodes(g, k=sample_size)
                mode_str = "Hub-based"

            print(f"Sampling {sample_size} nodes ({action} mode) with fanouts {sample_fanouts}...")

            plot_subgraph_with_neighbors(
                g,
                target_nodes=target_nodes,
                n_hop=len(sample_fanouts),
                fanouts=sample_fanouts,
                title=args.title or f"{mode_str} Preview of Graph {args.index}",
                edge_weight_name=args.edge_weight,
                node_palette=args.node_palette,
                edge_palette=args.edge_palette,
                reverse_etypes=args.reverse_etypes,
            )

        if args.output:
            plt.savefig(args.output)
            print(f"Saved preview to {args.output}")
        else:
            plt.show()

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("Hint: Please specify a valid Seaborn/Matplotlib palette name.", file=sys.stderr)
        return


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
    view_parser.add_argument("--reverse-etypes", type=str, help="Reverse edge types as a JSON string")
    view_parser.add_argument("--force", "-f", action="store_true", help="Force rendering of large graphs without sampling")

    args = parser.parse_args()

    if args.command == "view":
        view(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
