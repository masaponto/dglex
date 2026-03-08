import argparse
import sys
from typing import List


def _format_graph_info(g: "dgl.DGLGraph", graphs_count: int) -> str:  # noqa: F821
    """グラフ情報を整形した文字列を返す pure function。

    Args:
        g (dgl.DGLGraph): 情報を表示する DGL グラフ。
        graphs_count (int): ファイル内のグラフ総数。

    Returns:
        str: 整形されたグラフ情報文字列。
    """
    lines: List[str] = []

    # Graph Summary
    graph_type = "homogeneous" if g.is_homogeneous else "heterogeneous"
    lines.append("Graph Summary")
    lines.append("-------------")
    lines.append(f"Graphs in file : {graphs_count}")
    lines.append(f"Graph type     : {graph_type}")

    # Nodes
    lines.append("")
    lines.append("Nodes")
    lines.append("-----")
    if g.is_homogeneous:
        lines.append(f"_N : {g.num_nodes():,}")
    else:
        for ntype in g.ntypes:
            lines.append(f"{ntype} : {g.num_nodes(ntype):,}")

    # Edges
    lines.append("")
    lines.append("Edges")
    lines.append("-----")
    if g.is_homogeneous:
        lines.append(f"_E : {g.num_edges():,}")
    else:
        for src_type, etype, dst_type in g.canonical_etypes:
            label = f"{src_type} -> {dst_type}"
            lines.append(f"{label} : {g.num_edges((src_type, etype, dst_type)):,}")

    # Node Features
    has_node_features = False
    if g.is_homogeneous:
        has_node_features = len(g.ndata) > 0
    else:
        has_node_features = any(len(g.nodes[ntype].data) > 0 for ntype in g.ntypes)

    if has_node_features:
        lines.append("")
        lines.append("Node Features")
        lines.append("-------------")
        if g.is_homogeneous:
            for feat_name, tensor in g.ndata.items():
                dtype = str(tensor.dtype).replace("torch.", "")
                shape = tuple(tensor.shape)
                lines.append(f"{feat_name} : {dtype} {shape}")
        else:
            for ntype in g.ntypes:
                for feat_name, tensor in g.nodes[ntype].data.items():
                    dtype = str(tensor.dtype).replace("torch.", "")
                    shape = tuple(tensor.shape)
                    lines.append(f"{ntype}.{feat_name} : {dtype} {shape}")

    # Edge Features
    has_edge_features = False
    if g.is_homogeneous:
        has_edge_features = len(g.edata) > 0
    else:
        has_edge_features = any(
            len(g.edges[etype].data) > 0 for etype in g.canonical_etypes
        )

    if has_edge_features:
        lines.append("")
        lines.append("Edge Features")
        lines.append("-------------")
        if g.is_homogeneous:
            for feat_name, tensor in g.edata.items():
                dtype = str(tensor.dtype).replace("torch.", "")
                shape = tuple(tensor.shape)
                lines.append(f"{feat_name} : {dtype} {shape}")
        else:
            for src_type, etype, dst_type in g.canonical_etypes:
                for feat_name, tensor in g.edges[(src_type, etype, dst_type)].data.items():
                    dtype = str(tensor.dtype).replace("torch.", "")
                    shape = tuple(tensor.shape)
                    label = f"{src_type}->{dst_type}.{feat_name}"
                    lines.append(f"{label} : {dtype} {shape}")

    return "\n".join(lines)


def info(args: argparse.Namespace) -> None:
    """info サブコマンドの実行関数。DGL グラフファイルの構造情報を表示する。

    Args:
        args (argparse.Namespace): コマンドライン引数。`path` 属性を持つ。
    """
    import dgl

    if not hasattr(args, "path") or not args.path:
        print("Error: path is required", file=sys.stderr)
        return

    if not sys.modules.get("os", __import__("os")).path.exists(args.path):
        print("Error: file not found", file=sys.stderr)
        return

    try:
        graphs, _ = dgl.load_graphs(args.path)
    except Exception:
        print("Error: file does not contain a valid DGL graph", file=sys.stderr)
        return

    if not graphs:
        print("Error: file does not contain a valid DGL graph", file=sys.stderr)
        return

    g = graphs[0]
    print(_format_graph_info(g, len(graphs)))
