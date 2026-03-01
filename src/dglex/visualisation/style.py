import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import dgl
import networkx as nx
from typing import Union, Any

# Type aliases
Color = Union[str, tuple[float, ...]]
Legend = plt.Line2D


def _get_colors(
    n: int, colors: Union[list[Color], None], palette_name: str = "tab10"
) -> list[Color]:
    if colors is None:
        colors = sns.color_palette(palette=palette_name, n_colors=n)
    return colors


def _get_colormap_from_base_color(base_color: Color) -> mcolors.LinearSegmentedColormap:
    light_color = sns.blend_palette(["white", base_color], as_cmap=False)[1]
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "custom_cmap", [light_color, base_color]
    )
    return cmap


def _get_colors_from_weights(
    base_color: Color, weight_list: list[float]
) -> list[Color]:
    if len(weight_list) == 0:
        return []

    min_weight = min(weight_list)
    max_weight = max(weight_list)

    if min_weight < 0:
        weight_list = [w - min_weight for w in weight_list]
        max_weight -= min_weight

    if max_weight > 1:
        weight_list = [w / max_weight for w in weight_list]

    cmap = _get_colormap_from_base_color(base_color)
    colors = [cmap(w) for w in weight_list]
    return colors


def _get_homogeneous_node_colors_and_legend(
    ntype_colors: list[Color],
) -> tuple[list[Color], list[Legend]]:
    node_colors = ntype_colors
    node_legend = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=ntype_colors[0],
            markersize=10,
            label="Nodes",
        )
    ]
    return node_colors, node_legend


def _get_heterogenous_node_colors_and_legend(
    hetero_graph: dgl.DGLHeteroGraph, ng: nx.Graph, ntype_colors: list[Color]
) -> tuple[list[Color], list[Legend]]:
    node_colors = [
        ntype_colors[ndata[1][dgl.NTYPE].item()] for ndata in ng.nodes(data=True)
    ]
    node_legend = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=ntype_colors[i],
            markersize=10,
            label=ntype,
        )
        for i, ntype in enumerate(hetero_graph.ntypes)
    ]
    return node_colors, node_legend


def _get_homogeneous_edge_colors_and_legend(
    etype_colors: list[Color], edge_weight: Any = None
) -> tuple[list[Color], list[Legend]]:
    if edge_weight is None:
        edge_colors = etype_colors
    else:
        edge_colors = _get_colors_from_weights(etype_colors[0], edge_weight.tolist())

    edge_legend = [
        plt.Line2D(
            [0],
            [0],
            lw=2,
            color=etype_colors[0],
            markerfacecolor=etype_colors[0],
            markersize=10,
            label="Edge",
        )
    ]
    return edge_colors, edge_legend


def _get_heterogeneous_edge_colors_and_legend(
    hetero_graph: dgl.DGLHeteroGraph,
    ng: nx.Graph,
    etype_colors: list[Color],
    edge_weight: Any = None,
) -> tuple[list[Color], list[Legend]]:
    if edge_weight is None:
        edge_colors = [
            etype_colors[edata[2][dgl.ETYPE].item()] for edata in ng.edges(data=True)
        ]
    else:
        _etype_colors_with_weight = [
            _get_colors_from_weights(
                etype_colors[i], edge_weight[canonical_etype].tolist()
            )
            for i, canonical_etype in enumerate(hetero_graph.canonical_etypes)
        ]

        edge_colors = [
            _etype_colors_with_weight[edata[2][dgl.ETYPE].item()][
                edata[2][dgl.EID].item()
            ]
            for edata in ng.edges(data=True)
        ]

    edge_legend = [
        plt.Line2D(
            [0],
            [0],
            lw=2,
            color=etype_colors[i],
            markerfacecolor=etype_colors[i],
            markersize=10,
            label=etype,
        )
        for i, etype in enumerate(hetero_graph.etypes)
    ]
    return edge_colors, edge_legend


def _get_heterogeneous_edge_colors_and_legend_reverse_etypes(
    hetero_graph: dgl.DGLHeteroGraph,
    ng: nx.Graph,
    etype_colors: list[Color],
    ueid_master: dict[int, Any],
    edge_weight: Any = None,
) -> tuple[list[Color], list[Legend]]:
    if edge_weight is None:
        edge_colors = [
            etype_colors[ueid_master[edata[2][dgl.ETYPE].item()].ueid]
            for edata in ng.edges(data=True)
        ]
    else:
        _etype_colors_with_weight = [
            _get_colors_from_weights(
                etype_colors[ueid_master[i].ueid], edge_weight[canonical_etype].tolist()
            )
            for i, canonical_etype in enumerate(hetero_graph.canonical_etypes)
        ]

        edge_colors = [
            _etype_colors_with_weight[edata[2][dgl.ETYPE].item()][
                edata[2][dgl.EID].item()
            ]
            for edata in ng.edges(data=True)
        ]

    edge_legend = []
    processed_ueid = []
    for i, etype in enumerate(hetero_graph.etypes):
        if ueid_master[i].ueid in processed_ueid:
            continue

        edge_legend.append(
            plt.Line2D(
                [0],
                [0],
                lw=2,
                color=etype_colors[ueid_master[i].ueid],
                markerfacecolor=etype_colors[ueid_master[i].ueid],
                markersize=10,
                label=ueid_master[i].etype_legend_name,
            )
        )
        processed_ueid.append(ueid_master[i].ueid)

    return edge_colors, edge_legend


def _get_homogenous_node_labels(
    graph: dgl.DGLGraph, ng: nx.Graph, node_labels: Union[dict[int, str], None]
) -> dict[int, str]:
    if node_labels is None:
        node_labels = {k: str(k) for k in ng.nodes()}

    if len(node_labels) != graph.number_of_nodes():
        raise ValueError(
            f"Not enough node labels, node_labels: {len(node_labels)}, graph: {graph.number_of_nodes()}"
        )
    return node_labels


def _get_heterogenous_node_labels(
    hetero_graph: dgl.DGLHeteroGraph,
    ng: nx.Graph,
    node_labels: Union[dict[str, dict[int, str]], None],
) -> dict[int, str]:
    ntypes = hetero_graph.ntypes

    if node_labels is None:
        _node_labels = {
            ndata[0]: f"{ntypes[ndata[1][dgl.NTYPE].item()]}_{ndata[1][dgl.NID].item()}"
            for ndata in ng.nodes(data=True)
        }
    else:
        _node_labels = {
            ndata[0]: node_labels[ntypes[ndata[1][dgl.NTYPE].item()]][
                ndata[1][dgl.NID].item()
            ]
            for ndata in ng.nodes(data=True)
        }

    if len(_node_labels) != hetero_graph.number_of_nodes():
        raise ValueError(
            f"Not enough node labels, node_labels: {len(_node_labels)}, graph: {hetero_graph.number_of_nodes()}"
        )
    return _node_labels


def _get_homogeneous_edge_labels(
    graph: dgl.DGLGraph, edge_weight: Any
) -> Union[dict[tuple[int, int], str], None]:
    if edge_weight is None:
        return None

    edge_labels = {
        (src.item(), dst.item()): f"{weight:.3f}"
        for src, dst, weight in zip(
            graph.edges()[0], graph.edges()[1], edge_weight.tolist()
        )
    }
    return edge_labels


def _get_heterogeneous_edge_labels(
    hetero_graph: dgl.DGLHeteroGraph,
    ng: nx.Graph,
    edge_weight: Any,
) -> Union[dict[tuple[int, int], str], None]:
    if edge_weight is None:
        return None

    _edge_weight = [
        edge_weight[etype].tolist() for etype in hetero_graph.canonical_etypes
    ]

    edge_labels = {
        (edata[0], edata[1]): f"{_weight:.3f}"
        for edata in ng.edges(data=True)
        for _weight in _edge_weight[edata[2][dgl.ETYPE].item()]
    }
    return edge_labels


def _get_heterogeneous_edge_labels_reverse_etypes(
    hetero_graph: dgl.DGLHeteroGraph,
    ng: nx.Graph,
    edge_weight: Any,
) -> Union[dict[tuple[int, int], str], None]:
    if edge_weight is None:
        return None

    _edge_weight = [
        edge_weight[etype].tolist() for etype in hetero_graph.canonical_etypes
    ]

    edge_labels = {
        (
            edata[0],
            edata[1],
        ): f"{_edge_weight[edata[2][dgl.ETYPE].item()][edata[2][dgl.EID].item()]:.3f}"
        for edata in ng.edges(data=True)
    }
    return edge_labels
