import argparse
import os
import sys

from dglex.info.backends import UnsupportedGraphError, create_graph_info_adapter
from dglex.info.loaders import GraphLoadError, load_graph_from_path
from dglex.info.types import GraphInfo


def _format_graph_info(graph_info: GraphInfo) -> str:
    """GraphInfo を CLI 表示用テキストへ整形する。

    Args:
        graph_info: 整形対象のグラフ情報。

    Returns:
        str: CLI と同じ形式の整形済み文字列。
    """
    lines: list[str] = []

    lines.append("Graph Summary")
    lines.append("-------------")
    lines.append(f"Graphs in file : {graph_info.graphs_count}")
    lines.append(f"Graph type     : {graph_info.graph_type}")

    lines.append("")
    lines.append("Nodes")
    lines.append("-----")
    for label, count in graph_info.num_nodes.items():
        display_count = "not defined" if count is None else f"{count:,}"
        lines.append(f"{label} : {display_count}")

    lines.append("")
    lines.append("Edges")
    lines.append("-----")
    for label, count in graph_info.num_edges.items():
        display_label = label.replace("->", " -> ")
        lines.append(f"{display_label} : {count:,}")

    lines.append("")
    lines.append("Degree Statistics")
    lines.append("-----------------")
    for ntype, stats in graph_info.degree_stats.items():
        lines.append(
            f"{ntype} in  : mean={stats.in_degree.mean:.2f} "
            f"median={stats.in_degree.median:.2f} "
            f"min={stats.in_degree.min:.2f} max={stats.in_degree.max:.2f}"
        )
        lines.append(
            f"{ntype} out : mean={stats.out_degree.mean:.2f} "
            f"median={stats.out_degree.median:.2f} "
            f"min={stats.out_degree.min:.2f} max={stats.out_degree.max:.2f}"
        )

    if graph_info.node_features:
        lines.append("")
        lines.append("Node Features")
        lines.append("-------------")
        for feat_name, feature in graph_info.node_features.items():
            lines.append(f"{feat_name} : {feature.dtype} {feature.shape}")

    if graph_info.edge_features:
        lines.append("")
        lines.append("Edge Features")
        lines.append("-------------")
        for feat_name, feature in graph_info.edge_features.items():
            lines.append(f"{feat_name} : {feature.dtype} {feature.shape}")

    if graph_info.warnings:
        lines.append("")
        lines.append("Warnings")
        lines.append("--------")
        for warning in graph_info.warnings:
            lines.append(f"- {warning}")

    return "\n".join(lines)


def get_graph_info(graph: object, graphs_count: int = 1) -> GraphInfo:
    """グラフオブジェクトから backend 非依存の GraphInfo を構築して返す。

    Args:
        graph: DGL または PyG のグラフオブジェクト。
        graphs_count: ファイル内のグラフ総数。

    Returns:
        GraphInfo: 構造化されたグラフ情報。

    Raises:
        UnsupportedGraphError: サポート対象外のグラフだった場合。
    """
    adapter = create_graph_info_adapter(graph, graphs_count)
    graph_info = adapter.to_graph_info()
    graph_info.summary = _format_graph_info(graph_info)
    return graph_info


def info(args: argparse.Namespace) -> None:
    """info サブコマンドを実行する。

    Args:
        args: コマンドライン引数。`path` 属性を持つ。
    """
    if not hasattr(args, "path") or not args.path:
        print("Error: path is required", file=sys.stderr)
        return

    if not os.path.exists(args.path):
        print("Error: file not found", file=sys.stderr)
        return

    try:
        loaded_graph = load_graph_from_path(args.path)
        graph_info = get_graph_info(loaded_graph.graph, loaded_graph.graphs_count)
    except (GraphLoadError, UnsupportedGraphError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return

    print(graph_info.summary)
