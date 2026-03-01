import dgl
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
from typing import Union

from dglex.visualisation.style import (
    Color,
    Legend,
    _get_colors,
    _get_homogeneous_node_colors_and_legend,
    _get_heterogenous_node_colors_and_legend,
    _get_homogeneous_edge_colors_and_legend,
    _get_heterogeneous_edge_colors_and_legend,
    _get_heterogeneous_edge_colors_and_legend_reverse_etypes,
    _get_homogenous_node_labels,
    _get_heterogenous_node_labels,
    _get_homogeneous_edge_labels,
    _get_heterogeneous_edge_labels,
    _get_heterogeneous_edge_labels_reverse_etypes,
)
from dglex.visualisation.graph_ops import (
    _get_edge_weight,
    _validate_edge_weight,
    _create_ueid_master,
    _count_heterogeneous_edges,
    _validate_target_nodes,
    _extract_subgraph,
)


def _plot(
    nx_g: nx.Graph,
    title: str = "",
    node_labels: Union[dict[int, str], None] = None,
    node_colors: Union[list[Color], None] = None,
    node_legend: Union[list[Legend], None] = None,
    edge_labels: Union[dict[tuple[int, int], float], None] = None,
    edge_colors: Union[list[Color], None] = None,
    edge_legend: Union[list[Legend], None] = None,
    figsize: tuple[int, int] = (10, 10),
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
        edge_color=edge_colors,
        **kwargs,
    )

    if edge_labels is not None:
        nx.draw_networkx_edge_labels(nx_g, pos, edge_labels=edge_labels)

    plt.legend(handles=node_legend + edge_legend, loc="best")
    ax.set_title(title)

    return ax


def _process_homogeneous_graph(
    graph,
    node_labels,
    ntype_colors,
    etype_colors,
    edge_weight_name,
    node_palette="tab10",
    edge_palette="tab10",
):
    if graph.num_nodes() == 0:
        raise ValueError(f"graph must have nodes, {graph}")

    node_attrs = [dgl.NID] if dgl.NID in graph.ndata else []
    edge_attrs = [dgl.EID] if dgl.EID in graph.edata else []
    ng = dgl.to_networkx(graph, node_attrs=node_attrs, edge_attrs=edge_attrs)
    node_labels = _get_homogenous_node_labels(graph, ng, node_labels)
    ntype_colors = _get_colors(1, ntype_colors, palette_name=node_palette)
    etype_colors = _get_colors(1, etype_colors, palette_name=edge_palette)
    edge_weight = _get_edge_weight(graph, edge_weight_name)
    _validate_edge_weight(
        edge_weight, edge_weight.shape if edge_weight is not None else None
    )
    edge_labels = _get_homogeneous_edge_labels(graph, edge_weight)
    node_colors, node_legend = _get_homogeneous_node_colors_and_legend(ntype_colors)
    edge_colors, edge_legend = _get_homogeneous_edge_colors_and_legend(
        etype_colors, edge_weight
    )
    return (
        ng,
        node_labels,
        node_colors,
        node_legend,
        edge_labels,
        edge_colors,
        edge_legend,
    )


def _process_heterogeneous_graph(
    graph,
    node_labels,
    ntype_colors,
    etype_colors,
    edge_weight_name,
    reverse_etypes,
    node_palette="tab10",
    edge_palette="tab10",
):
    homo_graph = dgl.to_homogeneous(graph, store_type=True)
    node_attrs = [dgl.NID, dgl.NTYPE]
    edge_attrs = [dgl.EID, dgl.ETYPE]
    ng = dgl.to_networkx(homo_graph, node_attrs=node_attrs, edge_attrs=edge_attrs)
    node_labels = _get_heterogenous_node_labels(graph, ng, node_labels)
    ntype_colors = _get_colors(
        len(graph.ntypes), ntype_colors, palette_name=node_palette
    )
    node_colors, node_legend = _get_heterogenous_node_colors_and_legend(
        graph, ng, ntype_colors
    )

    if reverse_etypes is None:
        etype_colors = _get_colors(
            len(graph.etypes), etype_colors, palette_name=edge_palette
        )
        edge_weight = _get_edge_weight(graph, edge_weight_name)
        if edge_weight is not None:
            for _, _weight in edge_weight.items():
                _validate_edge_weight(_weight, _weight.shape)
        edge_labels = _get_heterogeneous_edge_labels(graph, ng, edge_weight)
        edge_colors, edge_legend = _get_heterogeneous_edge_colors_and_legend(
            graph, ng, etype_colors, edge_weight
        )
    else:
        for etype in reverse_etypes:
            if etype not in reverse_etypes.values():
                raise ValueError(
                    f"etype {etype} was not found in reverse_etypes. reverse_etypes must contain all edge types."
                )
        etype_colors = _get_colors(
            _count_heterogeneous_edges(graph, reverse_etypes),
            etype_colors,
            palette_name=edge_palette,
        )
        edge_weight = _get_edge_weight(graph, edge_weight_name)
        if edge_weight is not None:
            for _, _weight in edge_weight.items():
                _validate_edge_weight(_weight, _weight.shape)
        ueid_master = _create_ueid_master(graph, reverse_etypes)
        edge_labels = _get_heterogeneous_edge_labels_reverse_etypes(
            graph, ng, edge_weight
        )
        edge_colors, edge_legend = (
            _get_heterogeneous_edge_colors_and_legend_reverse_etypes(
                graph, ng, etype_colors, ueid_master, edge_weight
            )
        )

    return (
        ng,
        node_labels,
        node_colors,
        node_legend,
        edge_labels,
        edge_colors,
        edge_legend,
    )


def plot_graph(
    graph: Union[dgl.DGLGraph, dgl.DGLHeteroGraph],
    title: str = "",
    node_labels: Union[dict[int, str], dict[str, dict[int, str]], None] = None,
    edge_weight_name: Union[str, None] = None,
    ntype_colors: Union[list[Color], None] = None,
    etype_colors: Union[list[Color], None] = None,
    node_palette: str = "tab10",
    edge_palette: str = "tab10",
    figsize: tuple[int, int] = (6, 4),
    reverse_etypes: Union[dict[str, str], None] = None,
    **kwargs,
) -> matplotlib.axes.Axes:
    """
    Plots a DGL graph using NetworkX and Matplotlib.
    """
    if graph.is_homogeneous:
        (
            ng,
            node_labels,
            node_colors,
            node_legend,
            edge_labels,
            edge_colors,
            edge_legend,
        ) = _process_homogeneous_graph(
            graph,
            node_labels,
            ntype_colors,
            etype_colors,
            edge_weight_name,
            node_palette,
            edge_palette,
        )
    else:
        (
            ng,
            node_labels,
            node_colors,
            node_legend,
            edge_labels,
            edge_colors,
            edge_legend,
        ) = _process_heterogeneous_graph(
            graph,
            node_labels,
            ntype_colors,
            etype_colors,
            edge_weight_name,
            reverse_etypes,
            node_palette,
            edge_palette,
        )

    return _plot(
        ng,
        title=title,
        node_labels=node_labels,
        node_colors=node_colors,
        node_legend=node_legend,
        edge_labels=edge_labels,
        edge_colors=edge_colors,
        edge_legend=edge_legend,
        figsize=figsize,
        **kwargs,
    )


def plot_subgraph_with_neighbors(
    graph: dgl.DGLGraph,
    target_nodes: Union[list[int], dict[str, list[int]]],
    n_hop: int,
    fanouts: Union[list[int], list[dict[str, int]], None] = None,
    title: str = "",
    node_labels: Union[dict[int, str], dict[str, dict[int, str]], None] = None,
    edge_weight_name: Union[str, None] = None,
    ntype_colors: Union[list[Color], None] = None,
    etype_colors: Union[list[Color], None] = None,
    figsize: tuple[int, int] = (6, 4),
    reverse_etypes: Union[dict[str, str], None] = None,
    **kwargs,
) -> matplotlib.axes.Axes:
    """
    Plots a subgraph centered around specified target nodes, including their neighbors up to a certain hop distance.
    """
    if n_hop <= 0:
        raise ValueError(
            f"A argument n_hop must be greater than 0. The current n_hop is {n_hop}"
        )

    if fanouts is not None and len(fanouts) != n_hop:
        raise ValueError(
            f"The fanouts must have the same length as n_hop. The current fanouts is {fanouts}"
        )

    _validate_target_nodes(graph, target_nodes)

    sg, node_labels = _extract_subgraph(
        graph, target_nodes, n_hop, fanouts, node_labels, edge_weight_name
    )

    return plot_graph(
        sg,
        title=title,
        node_labels=node_labels,
        ntype_colors=ntype_colors,
        etype_colors=etype_colors,
        edge_weight_name=edge_weight_name,
        figsize=figsize,
        reverse_etypes=reverse_etypes,
        **kwargs,
    )
