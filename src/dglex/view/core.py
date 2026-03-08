import argparse
import sys

from dglex.view.config import load_config, merge_config
from dglex.view.types import SamplingAction


def _decide_sampling_action(
    num_nodes: int,
    num_edges: int,
    limits: dict,
    force: bool,
    is_tty: bool,
    has_output: bool,
    user_choice: str,
) -> SamplingAction:
    """グラフサイズとユーザー選択からアクションを決定する純粋関数（I/O なし）。

    Args:
        num_nodes (int): グラフのノード数。
        num_edges (int): グラフのエッジ数。
        limits (dict): max_nodes, max_edges を含む制限設定。
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


def view(args: argparse.Namespace) -> None:
    """view サブコマンドを実行する。大規模グラフへの安全チェックを含む。

    Args:
        args (argparse.Namespace): コマンドライン引数。
    """
    import dgl
    import matplotlib.pyplot as plt

    from dglex.view.graph_ops import get_random_nodes, get_top_k_degree_nodes
    from dglex.view.plot import plot_graph, plot_subgraph_with_neighbors

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
        print(
            f"Index {args.index} is out of range. The file contains {len(graphs)} graphs.",
            file=sys.stderr,
        )
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
            print(
                "Action: Automatic random sampling applied for safety. Use --force to override.",
                file=sys.stderr,
            )

    action = _decide_sampling_action(
        num_nodes,
        num_edges,
        limits,
        args.force,
        sys.stdin.isatty(),
        bool(args.output),
        user_choice,
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
            else:  # hub
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
