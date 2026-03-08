import argparse
import statistics
import sys
from typing import TYPE_CHECKING, Dict, List

from dglex.info.types import DegreeStats, FeatureInfo, GraphInfo, NodeDegreeStats

if TYPE_CHECKING:
    import dgl


def _build_degree_stats(values: List[int]) -> DegreeStats:
    """次数配列から統計情報を算出する。"""
    if not values:
        return DegreeStats(mean=0.0, median=0.0, min=0.0, max=0.0)

    mean = float(sum(values)) / len(values)
    return DegreeStats(
        mean=mean,
        median=float(statistics.median(values)),
        min=float(min(values)),
        max=float(max(values)),
    )


def _compute_degree_stats(g: "dgl.DGLGraph") -> Dict[str, NodeDegreeStats]:
    """ノード種別ごとの in/out 次数統計を返す。"""
    degree_stats: Dict[str, NodeDegreeStats] = {}

    if g.is_homogeneous:
        in_vals = g.in_degrees().tolist()
        out_vals = g.out_degrees().tolist()
        degree_stats["_N"] = NodeDegreeStats(
            in_degree=_build_degree_stats(in_vals),
            out_degree=_build_degree_stats(out_vals),
        )
        return degree_stats

    for ntype in g.ntypes:
        num_nodes = g.num_nodes(ntype)
        in_degree_values = [0] * num_nodes
        out_degree_values = [0] * num_nodes

        for src_type, etype, dst_type in g.canonical_etypes:
            if dst_type == ntype:
                for idx, degree in enumerate(g.in_degrees(etype=(src_type, etype, dst_type)).tolist()):
                    in_degree_values[idx] += int(degree)
            if src_type == ntype:
                for idx, degree in enumerate(g.out_degrees(etype=(src_type, etype, dst_type)).tolist()):
                    out_degree_values[idx] += int(degree)

        degree_stats[ntype] = NodeDegreeStats(
            in_degree=_build_degree_stats(in_degree_values),
            out_degree=_build_degree_stats(out_degree_values),
        )

    return degree_stats


def _format_graph_info(g: "dgl.DGLGraph", graphs_count: int) -> str:
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

    degree_stats = _compute_degree_stats(g)
    lines.append("")
    lines.append("Degree Statistics")
    lines.append("-----------------")
    for ntype, stats in degree_stats.items():
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


def get_graph_info(g: "dgl.DGLGraph", graphs_count: int = 1) -> GraphInfo:
    """DGL グラフから GraphInfo を構築して返す。

    Args:
        g (dgl.DGLGraph): 情報を取得する DGL グラフ。
        graphs_count (int): ファイル内のグラフ総数（デフォルト 1）。

    Returns:
        GraphInfo: 構造化されたグラフ情報。
    """
    graph_type = "homogeneous" if g.is_homogeneous else "heterogeneous"

    # num_nodes
    num_nodes: Dict[str, int]
    if g.is_homogeneous:
        num_nodes = {"_N": g.num_nodes()}
    else:
        num_nodes = {ntype: g.num_nodes(ntype) for ntype in g.ntypes}

    # num_edges
    num_edges: Dict[str, int]
    if g.is_homogeneous:
        num_edges = {"_E": g.num_edges()}
    else:
        num_edges = {
            f"{src_type}->{dst_type}": g.num_edges((src_type, etype, dst_type))
            for src_type, etype, dst_type in g.canonical_etypes
        }

    # node_features
    node_features: Dict[str, FeatureInfo] = {}
    if g.is_homogeneous:
        for feat_name, tensor in g.ndata.items():
            dtype = str(tensor.dtype).replace("torch.", "")
            node_features[feat_name] = FeatureInfo(dtype=dtype, shape=tuple(tensor.shape))
    else:
        for ntype in g.ntypes:
            for feat_name, tensor in g.nodes[ntype].data.items():
                dtype = str(tensor.dtype).replace("torch.", "")
                node_features[f"{ntype}.{feat_name}"] = FeatureInfo(
                    dtype=dtype, shape=tuple(tensor.shape)
                )

    # edge_features
    edge_features: Dict[str, FeatureInfo] = {}
    if g.is_homogeneous:
        for feat_name, tensor in g.edata.items():
            dtype = str(tensor.dtype).replace("torch.", "")
            edge_features[feat_name] = FeatureInfo(dtype=dtype, shape=tuple(tensor.shape))
    else:
        for src_type, etype, dst_type in g.canonical_etypes:
            for feat_name, tensor in g.edges[(src_type, etype, dst_type)].data.items():
                dtype = str(tensor.dtype).replace("torch.", "")
                edge_features[f"{src_type}->{dst_type}.{feat_name}"] = FeatureInfo(
                    dtype=dtype, shape=tuple(tensor.shape)
                )

    degree_stats = _compute_degree_stats(g)
    summary = _format_graph_info(g, graphs_count)

    return GraphInfo(
        graph_type=graph_type,
        graphs_count=graphs_count,
        num_nodes=num_nodes,
        num_edges=num_edges,
        node_features=node_features,
        edge_features=edge_features,
        degree_stats=degree_stats,
        summary=summary,
    )


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
