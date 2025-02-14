import dgl
import matplotlib
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import networkx as nx
import seaborn as sns


def _plot(
    nx_g: nx.Graph,
    title: str = "",
    node_labels: Dict[int, str] = None,
    edge_labels: Dict[int, str] = None,
    node_colors: List[str] = None,
    node_legend: List[plt.Line2D] = None,
    figsize: Tuple[int, int] = (10, 10),
    *args,
    **kwargs,
) -> matplotlib.axes.Axes:
    fig, ax = plt.subplots(figsize=figsize)
    pos = nx.spring_layout(nx_g)

    nx.draw(
        nx_g,
        pos,
        ax=ax,
        labels=node_labels,
        node_color=node_colors,
        # edge_labels=edge_labels,
        *args,
        **kwargs,
    )

    plt.legend(handles=node_legend, loc="best")
    ax.set_title(title)

    return ax


def _get_homogenous_node_labels(
    graph: dgl.DGLGraph, ng: nx.Graph, node_labels: Dict[int, str]
) -> Dict[int, str]:
    if node_labels is None:
        node_labels = {k: str(k) for k in ng.nodes()}
    assert len(node_labels) == graph.number_of_nodes(), "Not enough node labels"
    return node_labels


def _get_heterogenous_node_labels(
    hetero_graph: dgl.DGLHeteroGraph, ng: nx.Graph, node_labels: Dict[int, str]
) -> Dict[int, str]:
    if node_labels is None:
        node_labels = {
            ndata[0]: f"{ndata[1][dgl.NTYPE].item()}_{ndata[1][dgl.NID].item()}"
            for ndata in ng.nodes(data=True)
        }
    assert len(node_labels) == hetero_graph.number_of_nodes(), "Not enough node labels"
    return node_labels


def _get_ntype_colors(n: int, ntype_colors: List[str]) -> List[str]:
    if ntype_colors is None:
        ntype_colors = sns.color_palette("tab10", n)
    return ntype_colors


def _get_homogeneous_node_colors_and_legend(
    ntype_colors: List[str],
) -> Tuple[List[str], List[plt.Line2D]]:
    node_colors = ntype_colors[0]
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
    hetero_graph: dgl.DGLHeteroGraph, ng: nx.Graph, ntype_colors: List[str]
) -> Tuple[List[str], List[plt.Line2D]]:
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


def _plot_graph(
    graph: dgl.DGLGraph,
    title: str = "",
    node_labels: Dict[int, str] = None,
    edge_labels: Dict[int, str] = None,
    ntype_colors: List[str] = None,
    figsize: Tuple[int, int] = (6, 4),
    *args,
    **kwargs,
) -> matplotlib.axes.Axes:

    if graph.is_homogeneous:
        node_attrs = [dgl.NID] if dgl.NID in graph.ndata else []
        edge_attrs = [dgl.EID] if dgl.EID in graph.edata else []
        ng = dgl.to_networkx(graph, node_attrs=node_attrs, edge_attrs=edge_attrs)
        node_labels = _get_homogenous_node_labels(graph, ng, node_labels)
        ntype_colors = _get_ntype_colors(1, ntype_colors)
        node_colors, node_legend = _get_homogeneous_node_colors_and_legend(ntype_colors)

    else:
        # heterogeneous graph
        homo_graph = dgl.to_homogeneous(graph, store_type=True)
        node_attrs = [dgl.NID, dgl.NTYPE]
        edge_attrs = [dgl.EID, dgl.ETYPE]
        ng = dgl.to_networkx(homo_graph, node_attrs=node_attrs, edge_attrs=edge_attrs)
        node_labels = _get_heterogenous_node_labels(graph, ng, node_labels)
        ntype_colors = _get_ntype_colors(len(graph.ntypes), ntype_colors)
        node_colors, node_legend = _get_heterogenous_node_colors_and_legend(
            graph, ng, ntype_colors
        )

    return _plot(
        ng,
        title=title,
        node_labels=node_labels,
        edge_labels=edge_labels,
        node_colors=node_colors,
        node_legend=node_legend,
        figsize=figsize,
        *args,
        **kwargs,
    )


def plot_subgraph_with_neighbors(
    graph: dgl.DGLGraph,
    node_labels: List[int],
    target_nid: int,
    n_hop: int,
    plot_args,
) -> matplotlib.axes.Axes:

    pass
