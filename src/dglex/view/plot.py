from typing import Optional, Union

import dgl
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx

from dglex.view.graph_ops import (
    _count_heterogeneous_edges,
    _create_ueid_master,
    _extract_subgraph,
    _get_edge_weight,
    _validate_edge_weight,
    _validate_target_nodes,
)
from dglex.view.style import (
    _get_colors,
    _get_heterogeneous_edge_colors_and_legend,
    _get_heterogeneous_edge_colors_and_legend_reverse_etypes,
    _get_heterogeneous_edge_labels,
    _get_heterogeneous_node_colors_and_legend,
    _get_heterogeneous_node_labels,
    _get_homogeneous_edge_colors_and_legend,
    _get_homogeneous_edge_labels,
    _get_homogeneous_node_colors_and_legend,
    _get_homogeneous_node_labels,
)
from dglex.view.types import Color, Legend, PlotConfig


def _plot(
    nx_g: nx.Graph,
    config: PlotConfig,
    node_labels: Optional[dict[int, str]] = None,
    node_colors: Optional[list[Color]] = None,
    node_legend: Optional[list[Legend]] = None,
    edge_labels: Optional[dict[tuple[int, int], str]] = None,
    edge_colors: Optional[list[Color]] = None,
    edge_legend: Optional[list[Legend]] = None,
    **kwargs,
) -> matplotlib.axes.Axes:
    fig, ax = plt.subplots(figsize=config.figsize)
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

    plt.legend(handles=(node_legend or []) + (edge_legend or []), loc="best")
    ax.set_title(config.title)

    return ax


def _process_homogeneous_graph(
    graph: dgl.DGLGraph,
    config: PlotConfig,
):
    if graph.num_nodes() == 0:
        raise ValueError(f"graph must have nodes, {graph}")

    node_attrs = [dgl.NID] if dgl.NID in graph.ndata else []
    edge_attrs = [dgl.EID] if dgl.EID in graph.edata else []
    ng = dgl.to_networkx(graph, node_attrs=node_attrs, edge_attrs=edge_attrs)

    node_labels = _get_homogeneous_node_labels(graph, ng, config.node_labels)  # type: ignore[arg-type]
    ntype_colors = _get_colors(1, config.ntype_colors, palette_name=config.node_palette)
    etype_colors = _get_colors(1, config.etype_colors, palette_name=config.edge_palette)

    edge_weight = _get_edge_weight(graph, config.edge_weight_name)
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
    graph: dgl.DGLHeteroGraph,
    config: PlotConfig,
):
    homo_graph = dgl.to_homogeneous(graph, store_type=True)
    node_attrs = [dgl.NID, dgl.NTYPE]
    edge_attrs = [dgl.EID, dgl.ETYPE]
    ng = dgl.to_networkx(homo_graph, node_attrs=node_attrs, edge_attrs=edge_attrs)

    node_labels = _get_heterogeneous_node_labels(graph, ng, config.node_labels)  # type: ignore[arg-type]
    ntype_colors = _get_colors(
        len(graph.ntypes), config.ntype_colors, palette_name=config.node_palette
    )
    node_colors, node_legend = _get_heterogeneous_node_colors_and_legend(
        graph, ng, ntype_colors
    )

    if config.reverse_etypes is None:
        etype_colors = _get_colors(
            len(graph.etypes), config.etype_colors, palette_name=config.edge_palette
        )
        edge_weight = _get_edge_weight(graph, config.edge_weight_name)
        if edge_weight is not None:
            for _, _weight in edge_weight.items():
                _validate_edge_weight(_weight, _weight.shape)

        edge_labels = _get_heterogeneous_edge_labels(graph, ng, edge_weight)
        edge_colors, edge_legend = _get_heterogeneous_edge_colors_and_legend(
            graph, ng, etype_colors, edge_weight
        )
    else:
        for etype in config.reverse_etypes:
            if etype not in config.reverse_etypes.values():
                raise ValueError(
                    f"etype {etype} was not found in reverse_etypes. reverse_etypes must contain all edge types."
                )

        etype_colors = _get_colors(
            _count_heterogeneous_edges(graph, config.reverse_etypes),
            config.etype_colors,
            palette_name=config.edge_palette,
        )
        edge_weight = _get_edge_weight(graph, config.edge_weight_name)
        if edge_weight is not None:
            for _, _weight in edge_weight.items():
                _validate_edge_weight(_weight, _weight.shape)

        ueid_master = _create_ueid_master(graph, config.reverse_etypes)
        edge_labels = _get_heterogeneous_edge_labels(graph, ng, edge_weight)
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
    node_labels: Optional[Union[dict[int, str], dict[str, dict[int, str]]]] = None,
    edge_weight_name: Optional[str] = None,
    ntype_colors: Optional[list[Color]] = None,
    etype_colors: Optional[list[Color]] = None,
    node_palette: str = "tab10",
    edge_palette: str = "tab10",
    figsize: tuple[int, int] = (6, 4),
    reverse_etypes: Optional[dict[str, str]] = None,
    **kwargs,
) -> matplotlib.axes.Axes:
    """
    Plots a DGL graph using NetworkX and Matplotlib.

    Args:
        graph (Union[dgl.DGLGraph, dgl.DGLHeteroGraph]): The DGL graph to plot.
        title (str, optional): The title of the plot. Defaults to "".
        node_labels (Optional[Union[dict[int, str], dict[str, dict[int, str]]]]): Node labels.
        edge_weight_name (Optional[str]): Name of the edge weight data field.
        ntype_colors (Optional[list[Color]]): Colors for node types.
        etype_colors (Optional[list[Color]]): Colors for edge types.
        node_palette (str, optional): Color palette for node types. Defaults to "tab10".
        edge_palette (str, optional): Color palette for edge types. Defaults to "tab10".
        figsize (tuple[int, int], optional): Figure size. Defaults to (6, 4).
        reverse_etypes (Optional[dict[str, str]]): Reverse edge types for bidirectional edges.
        **kwargs: Additional arguments for `nx.draw`.

    Returns:
        matplotlib.axes.Axes: The matplotlib axes object.
    """
    config = PlotConfig(
        title=title,
        figsize=figsize,
        node_palette=node_palette,
        edge_palette=edge_palette,
        node_labels=node_labels,
        edge_weight_name=edge_weight_name,
        ntype_colors=ntype_colors,
        etype_colors=etype_colors,
        reverse_etypes=reverse_etypes,
    )

    if graph.is_homogeneous:
        (
            ng,
            processed_node_labels,
            node_colors,
            node_legend,
            edge_labels,
            edge_colors,
            edge_legend,
        ) = _process_homogeneous_graph(graph, config)
    else:
        (
            ng,
            processed_node_labels,
            node_colors,
            node_legend,
            edge_labels,
            edge_colors,
            edge_legend,
        ) = _process_heterogeneous_graph(graph, config)

    return _plot(
        ng,
        config=config,
        node_labels=processed_node_labels,
        node_colors=node_colors,
        node_legend=node_legend,
        edge_labels=edge_labels,
        edge_colors=edge_colors,
        edge_legend=edge_legend,
        **kwargs,
    )


def plot_subgraph_with_neighbors(
    graph: dgl.DGLGraph,
    target_nodes: Union[list[int], dict[str, list[int]]],
    n_hop: int,
    fanouts: Optional[Union[list[int], list[dict[str, int]]]] = None,
    title: str = "",
    node_labels: Optional[Union[dict[int, str], dict[str, dict[int, str]]]] = None,
    edge_weight_name: Optional[str] = None,
    ntype_colors: Optional[list[Color]] = None,
    etype_colors: Optional[list[Color]] = None,
    node_palette: str = "tab10",
    edge_palette: str = "tab10",
    figsize: tuple[int, int] = (6, 4),
    reverse_etypes: Optional[dict[str, str]] = None,
    **kwargs,
) -> matplotlib.axes.Axes:
    """
    Plots a subgraph centered around target nodes.
    """
    if n_hop <= 0:
        raise ValueError(f"n_hop must be greater than 0. Got {n_hop}")

    if fanouts is not None and len(fanouts) != n_hop:
        raise ValueError(f"fanouts length must equal n_hop. fanouts: {fanouts}, n_hop: {n_hop}")

    _validate_target_nodes(graph, target_nodes)

    sg, sampled_node_labels = _extract_subgraph(
        graph, target_nodes, n_hop, fanouts, node_labels, edge_weight_name
    )

    return plot_graph(
        sg,
        title=title,
        node_labels=sampled_node_labels,
        ntype_colors=ntype_colors,
        etype_colors=etype_colors,
        edge_weight_name=edge_weight_name,
        node_palette=node_palette,
        edge_palette=edge_palette,
        figsize=figsize,
        reverse_etypes=reverse_etypes,
        **kwargs,
    )
