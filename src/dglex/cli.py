import argparse


def main() -> None:
    """dglex CLI のエントリポイント。サブコマンドを登録してディスパッチする。"""
    parser = argparse.ArgumentParser(description="dglex: DGL extension library CLI tool")
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # view sub-command
    view_parser = subparsers.add_parser("view", help="View a DGL graph file")
    view_parser.add_argument("path", type=str, help="Path to the DGL graph (.bin) file")
    view_parser.add_argument(
        "--index", "-i", type=int, default=0, help="Index of the graph to visualize (default: 0)"
    )
    view_parser.add_argument("--title", "-t", type=str, help="Title of the plot")
    view_parser.add_argument(
        "--edge-weight", "-w", type=str, help="Name of the edge weight data field"
    )
    view_parser.add_argument(
        "--node-palette",
        type=str,
        default="tab10",
        help="Seaborn color palette for nodes (default: tab10)",
    )
    view_parser.add_argument(
        "--edge-palette",
        type=str,
        default="tab10",
        help="Seaborn color palette for edges (default: tab10)",
    )
    view_parser.add_argument(
        "--output", "-o", type=str, help="Path to save the plot image (optional)"
    )
    view_parser.add_argument(
        "--figsize", type=int, nargs=2, help="Figure size as width height (optional)"
    )
    view_parser.add_argument(
        "--reverse-etypes", type=str, help="Reverse edge types as a JSON string"
    )
    view_parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force rendering of large graphs without sampling",
    )

    # info sub-command
    info_parser = subparsers.add_parser("info", help="Show graph file information")
    info_parser.add_argument(
        "path",
        type=str,
        help="Path to a serialized graph file (for example DGL .bin or PyG .pt)",
    )

    args = parser.parse_args()

    if args.command == "view":
        # Lazy import: avoid importing heavy visualization deps on CLI startup.
        from dglex.view.core import view

        view(args)
    elif args.command == "info":
        # Lazy import: keep startup lightweight unless info command is executed.
        from dglex.info.core import info

        info(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
