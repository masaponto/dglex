import dgl
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
import seaborn as sns

from typing import Union
from collections import defaultdict
from dataclasses import dataclass


# Type aliases
Color = Union[str, tuple[float]]
Legend = plt.Line2D


@dataclass
class EdgeTypeInfo:
    etype: str
    eid: int
    ueid: int
    etype_legend_name: str


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
        # edge_labels=edge_labels,
        **kwargs,
    )

    if edge_labels is not None:
        nx.draw_networkx_edge_labels(nx_g, pos, edge_labels=edge_labels)

    plt.legend(handles=node_legend + edge_legend, loc="best")
    ax.set_title(title)

    return ax


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


def _get_homogeneous_edge_labels(
    graph: dgl.DGLGraph, edge_weight: Union[torch.Tensor, None]
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


def _get_heterogeneous_edge_labels(
    hetero_graph: dgl.DGLHeteroGraph,
    ng: nx.Graph,
    edge_weight: Union[dict[tuple[str, str, str], torch.Tensor], None],
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


def _get_colors(
    n: int, colors: list[Color], palette_name: str = "tab10"
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
    etype_colors: list[Color], edge_weight: Union[torch.Tensor, None] = None
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
    edge_weight: Union[dict[tuple[str, str, str], torch.Tensor], None] = None,
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


def _get_heterogeneous_edge_labels_reverse_etypes(
    hetero_graph: dgl.DGLHeteroGraph,
    ng: nx.Graph,
    edge_weight: Union[dict[tuple[str, str, str], torch.Tensor], None],
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


def _get_heterogeneous_edge_colors_and_legend_reverse_etypes(
    hetero_graph: dgl.DGLHeteroGraph,
    ng: nx.Graph,
    etype_colors: list[Color],
    ueid_master: dict[int, EdgeTypeInfo],
    edge_weight: Union[dict[tuple[str, str, str], torch.Tensor], None] = None,
) -> tuple[list[Color], list[Legend]]:
    edge_colors = []
    # reverse_etypes_tuples = _get_revese_etypes_tuple(reverse_etypes)

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


def _count_heterogeneous_edges(
    graph: dgl.DGLHeteroGraph, reverse_etypes: dict[str, str]
) -> int:

    num_directed_edge = len(
        [etype for etype in graph.etypes if etype not in reverse_etypes]
    )
    num_undirected_edge = len(reverse_etypes) // 2

    return num_directed_edge + num_undirected_edge


def _create_ueid_master(
    hetero_graph: dgl.DGLHeteroGraph, reverse_etypes: dict[str, str]
) -> dict[int, EdgeTypeInfo]:
    """
    Define the same ID for an edge and its reversed edge as Unique Edge ID (ueid).
    """

    ueid_master = {}

    # create etype_eid_master
    # example: {'click': 0, 'clicked-by': 1, 'follow': 2, 'followed-by': 3}
    etype_eid_master = {etype: i for i, etype in enumerate(hetero_graph.etypes)}

    # Create reverse etype tuples
    # example: {'click': 'cilked-by', 'clicked-by':'click', 'follow': 'followed-by', 'followed-by': 'follow'}
    # => [('click', 'clicked-by'), ('follow', 'followed-by')]
    reverse_etype_tuples = []
    for k, v in reverse_etypes.items():
        if (v, k) not in reverse_etype_tuples:
            reverse_etype_tuples.append((k, v))

    # Create ueid_master
    # Here an edge and its reversed edge have the same ID
    # example: {0: EdgeTypeInfo('click', 0, 0, 'click-clikecd-by'),
    #           1: EdgeTypeInfo('clicked-by', 1, 0, 'click-clikecd-by')}
    #           2: EdgeTypeInfo('follow', 2, 1, 'follow-followed-by'),
    #           3: EdgeTypeInfo('followed-by', 3, 1, 'follow-followed-by')}
    for i, etype_tuple in enumerate(reverse_etype_tuples):
        ueid_master[etype_eid_master[etype_tuple[0]]] = EdgeTypeInfo(
            etype=etype_tuple[0],
            eid=etype_eid_master[etype_tuple[0]],
            ueid=i,
            etype_legend_name=etype_tuple[0] + "/" + etype_tuple[1],
        )
        ueid_master[etype_eid_master[etype_tuple[1]]] = EdgeTypeInfo(
            etype=etype_tuple[1],
            eid=etype_eid_master[etype_tuple[1]],
            ueid=i,
            etype_legend_name=etype_tuple[0] + "/" + etype_tuple[1],
        )

    # Add the remaining edge types that are directed
    directed_edges = [
        etype for etype in hetero_graph.etypes if etype not in reverse_etypes
    ]
    for i, _etype in enumerate(directed_edges):
        _ueid = i + len(reverse_etype_tuples)
        ueid_master[etype_eid_master[_etype]] = EdgeTypeInfo(
            etype=_etype,
            eid=etype_eid_master[_etype],
            ueid=_ueid,
            etype_legend_name=_etype,
        )

    return ueid_master


def _validate_edge_weight(edge_weight, shape):
    if edge_weight is not None and edge_weight.ndimension() > 1:
        raise ValueError(f"edge_weight must be 1D. The current shape is {shape}")


def _get_edge_weight(graph, edge_weight_name):
    return graph.edata[edge_weight_name] if edge_weight_name else None


def _process_homogeneous_graph(
    graph, node_labels, ntype_colors, etype_colors, edge_weight_name
):
    if graph.num_nodes() == 0:
        raise ValueError(f"graph must have nodes, {graph}")

    node_attrs = [dgl.NID] if dgl.NID in graph.ndata else []
    edge_attrs = [dgl.EID] if dgl.EID in graph.edata else []
    ng = dgl.to_networkx(graph, node_attrs=node_attrs, edge_attrs=edge_attrs)
    node_labels = _get_homogenous_node_labels(graph, ng, node_labels)
    ntype_colors = _get_colors(1, ntype_colors)
    etype_colors = _get_colors(1, etype_colors)
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
    graph, node_labels, ntype_colors, etype_colors, edge_weight_name, reverse_etypes
):
    homo_graph = dgl.to_homogeneous(graph, store_type=True)
    node_attrs = [dgl.NID, dgl.NTYPE]
    edge_attrs = [dgl.EID, dgl.ETYPE]
    ng = dgl.to_networkx(homo_graph, node_attrs=node_attrs, edge_attrs=edge_attrs)
    node_labels = _get_heterogenous_node_labels(graph, ng, node_labels)
    ntype_colors = _get_colors(len(graph.ntypes), ntype_colors)
    node_colors, node_legend = _get_heterogenous_node_colors_and_legend(
        graph, ng, ntype_colors
    )

    if reverse_etypes is None:
        etype_colors = _get_colors(len(graph.etypes), etype_colors)
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
            _count_heterogeneous_edges(graph, reverse_etypes), etype_colors
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
    figsize: tuple[int, int] = (6, 4),
    reverse_etypes: Union[dict[str, str], None] = None,
    **kwargs,
) -> matplotlib.axes.Axes:
    """
    Plots a DGL graph using NetworkX and Matplotlib.

    This function visualizes a DGL graph, handling both homogeneous and heterogeneous graphs.
    It can display node labels, edge weights, and different colors for node and edge types.
    For heterogeneous graphs, it can also handle reverse edge types to represent undirected edges.

    Args:
        graph (Union[dgl.DGLGraph, dgl.DGLHeteroGraph]): The DGL graph to plot.
        title (str, optional): The title of the plot. Defaults to "".
        node_labels (Union[dict[int, str], dict[str, dict[int, str]], None], optional):
            Node labels for the plot.
            For homogeneous graphs, this is a dictionary mapping node IDs to labels.
            For heterogeneous graphs, this is a dictionary where keys are node types and values are dictionaries mapping node IDs to labels.
            Defaults to None.
        edge_weight_name (Union[str, None], optional): The name of the edge data field to use for edge weights.
            If provided, edge colors will be based on these weights, and edge labels will display the weights. Defaults to None.
        ntype_colors (Union[list[Color], None], optional): A list of colors for node types.
            If None, a default color palette is used. Defaults to None.
        etype_colors (Union[list[Color], None], optional): A list of colors for edge types.
            If None, a default color palette is used. Defaults to None.
        figsize (tuple[int, int], optional): The figure size (width, height). Defaults to (6, 4).
        reverse_etypes (Union[dict[str, str], None], optional): A dictionary specifying reverse edge types for heterogeneous graphs.
            Keys are edge types, and values are the corresponding reverse edge types.
            Used for representing bidirectional edges. Defaults to None.
        **kwargs: Additional keyword arguments to pass to `nx.draw`.

    Returns:
        matplotlib.axes.Axes: The matplotlib axes object.

    Raises:
        ValueError: If the graph has no nodes, if edge_weight is not 1D, if `etype` was not found in reverse_etypes, if the target_nodes are not proper type for a given graph type.
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
            graph, node_labels, ntype_colors, etype_colors, edge_weight_name
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


def _extract_subgraph_nhop(
    g: dgl.DGLGraph,
    target_nodes: list[int],
    n_hop: int,
    node_labels: Union[dict[int, str], None] = None,
) -> tuple[dgl.DGLGraph, dict[int, str]]:

    seed_nodes = target_nodes
    for _i in range(n_hop):
        if _i == n_hop - 1:
            sg = dgl.in_subgraph(g, seed_nodes, relabel_nodes=True)
            break
        sg = dgl.in_subgraph(g, seed_nodes)
        src, dst = sg.edges()
        seed_nodes = torch.cat([src, dst]).unique()

    sampled_node_labels = {
        i: str(n) if node_labels is None else node_labels[n]
        for i, n in enumerate(sg.ndata[dgl.NID].tolist())
    }

    return sg, sampled_node_labels


def _extract_heterogeneous_subgraph_nhop(
    hetero_graph: dgl.DGLHeteroGraph,
    target_nodes: dict[str, list[int]],
    n_hop: int,
    node_labels: Union[dict[str, dict[int, str]], None] = None,
) -> tuple[dgl.DGLHeteroGraph, dict[int, str]]:
    seed_nodes = target_nodes
    for i in range(n_hop):

        if i == n_hop - 1:
            sg = dgl.in_subgraph(hetero_graph, seed_nodes, relabel_nodes=True)
            break

        sg = dgl.in_subgraph(hetero_graph, seed_nodes)

        seed_nodes = defaultdict(list)
        for src_ntype, etype, dst_ntype in hetero_graph.canonical_etypes:
            src, dst = sg.edges(etype=etype)
            seed_nodes[src_ntype].extend(src.tolist())
            seed_nodes[dst_ntype].extend(dst.tolist())
            # unique
            seed_nodes[src_ntype] = list(set(seed_nodes[src_ntype]))
            seed_nodes[dst_ntype] = list(set(seed_nodes[dst_ntype]))

    sampled_node_labels = {}
    for ntype in hetero_graph.ntypes:
        sampled_node_labels[ntype] = {
            i: f"{ntype}_{n}" if node_labels is None else node_labels[ntype][n]
            for i, n in enumerate(sg.ndata[dgl.NID][ntype].tolist())
        }

    return sg, sampled_node_labels


def _extract_sub_graph_nhop_fanouts(
    graph: dgl.DGLGraph,
    target_nodes: list[int],
    fanouts: list[int],
    node_labels: Union[dict[int, str], None],
    edge_weight_name: Union[str, None],
) -> tuple[dgl.DGLGraph, dict[int, str]]:

    sampler = dgl.dataloading.NeighborSampler(fanouts)
    dataloader = dgl.dataloading.DataLoader(
        graph,
        target_nodes,
        sampler,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=1,
    )

    with dataloader.enable_cpu_affinity():
        input_nodes, output_nodes, blocks = next(iter(dataloader))

    _src = []
    _dst = []
    _edge_weight = []
    sampled_node_labels = {}
    node_labels = {} if node_labels is None else node_labels
    for i, block in enumerate(blocks):
        src, dst = block.edges()
        _src.extend(src.tolist())
        _dst.extend(dst.tolist())

        if edge_weight_name is not None:
            ew = block.edata[edge_weight_name]
            _edge_weight.extend(ew.tolist())

    for i, nodes in enumerate(input_nodes.tolist()):
        sampled_node_labels[i] = (
            f"{nodes}" if i not in node_labels else node_labels[nodes]
        )

    sg = dgl.graph((_src, _dst))
    if edge_weight_name is not None:
        sg.edata[edge_weight_name] = torch.tensor(_edge_weight)

    return sg, sampled_node_labels


def _extract_heterogeneous_sub_graph_nhop_fanouts(
    graph: dgl.DGLHeteroGraph,
    target_nodes: dict[str, list[int]],
    fanouts: Union[list[int], list[dict[str, int]]],
    node_labels: Union[dict[str, dict[int, str]], None],
    edge_weight_name: Union[str, None],
) -> tuple[dgl.DGLHeteroGraph, dict[int, str]]:

    sampler = dgl.dataloading.NeighborSampler(fanouts)
    dataloader = dgl.dataloading.DataLoader(
        graph,
        target_nodes,
        sampler,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=1,
    )

    with dataloader.enable_cpu_affinity():
        input_nodes, output_nodes, blocks = next(iter(dataloader))

    graph_dic = {}
    edge_weight_dic = {}
    sampled_node_labels = {ntype: {} for ntype in graph.ntypes}
    node_labels = (
        {ntype: {} for ntype in graph.ntypes} if node_labels is None else node_labels
    )

    for src_ntype, etype, dst_ntype in graph.canonical_etypes:
        _src = []
        _dst = []
        _edge_weight = []
        for i, block in enumerate(blocks):
            src, dst = block.edges(etype=etype)
            _src.extend(src.tolist())
            _dst.extend(dst.tolist())

            if edge_weight_name is not None:
                ew = block.edata[edge_weight_name]
                _edge_weight.extend(ew[(src_ntype, etype, dst_ntype)].tolist())

        graph_dic[(src_ntype, etype, dst_ntype)] = (_src, _dst)

        if edge_weight_name is not None:
            edge_weight_dic[(src_ntype, etype, dst_ntype)] = torch.tensor(_edge_weight)

    for ntype in graph.ntypes:
        for i, nodes in enumerate(input_nodes[ntype].tolist()):
            sampled_node_labels[ntype][i] = (
                f"{nodes}" if i not in node_labels[ntype] else node_labels[ntype][nodes]
            )

    sg = dgl.heterograph(graph_dic)
    if edge_weight_name is not None:
        sg.edata[edge_weight_name] = edge_weight_dic

    return sg, sampled_node_labels


def _validate_target_nodes(
    graph: Union[dgl.DGLGraph, dgl.DGLHeteroGraph],
    target_nodes: Union[list[int], dict[str, list[int]]],
) -> None:
    if graph.is_homogeneous:
        if not isinstance(target_nodes, list):
            raise ValueError(
                f"The target_nodes must be a list for homogeneous graph. The current target_nodes is {target_nodes}"
            )
    else:
        if not isinstance(target_nodes, dict):
            raise ValueError(
                f"The target_nodes must be a dictionary for heterogeneous graph. The current target_nodes is {target_nodes}"
            )
        if any(not isinstance(value, list) for value in target_nodes.values()):
            raise ValueError(
                f"The target_nodes must be a dictionary of lists. The current target_nodes is {target_nodes}"
            )


def _extract_subgraph(
    graph: Union[dgl.DGLGraph, dgl.DGLHeteroGraph],
    target_nodes: Union[list[int], dict[str, list[int]]],
    n_hop: int,
    fanouts: Union[list[int], list[dict[str, int]], None],
    node_labels: Union[dict[int, str], dict[str, dict[int, str]], None],
    edge_weight_name: Union[str, None],
) -> tuple[dgl.DGLGraph, dict[int, str]]:

    if graph.is_homogeneous:
        if fanouts is None:
            return _extract_subgraph_nhop(graph, target_nodes, n_hop, node_labels)
        else:
            return _extract_sub_graph_nhop_fanouts(
                graph, target_nodes, fanouts, node_labels, edge_weight_name
            )
    else:
        if fanouts is None:
            return _extract_heterogeneous_subgraph_nhop(
                graph, target_nodes, n_hop, node_labels
            )
        else:
            return _extract_heterogeneous_sub_graph_nhop_fanouts(
                graph, target_nodes, fanouts, node_labels, edge_weight_name
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

    This function extracts a subgraph from the input graph, including the target nodes and their neighbors within a certain
    number of hops, and then plots it using the `plot_graph` function. It supports both homogeneous and heterogeneous graphs.
    You can control how the subgraph is extracted by specifying the number of hops (`n_hop`) or by using fanouts for neighbor sampling.

    Args:
        graph (dgl.DGLGraph): The input DGL graph (homogeneous or heterogeneous).
        target_nodes (Union[list[int], dict[str, list[int]]]): The target node(s) to center the subgraph around.
            For homogeneous graphs, this is a list of node IDs.
            For heterogeneous graphs, this is a dictionary where keys are node types and values are lists of node IDs.
        n_hop (int): The number of hops to include neighbors in the subgraph. Must be greater than 0.
        fanouts (Union[list[int], list[dict[str, int]], None], optional): The fanout for each hop when sampling neighbors.
            If None, all neighbors are included up to `n_hop`.
            For homogeneous graphs, this is a list of integers representing the number of neighbors to sample at each hop.
            For heterogeneous graphs, this is a list of dictionaries where keys are edge types and values are the number of neighbors to sample at each hop.
            The length of fanouts must be the same as n_hop. Defaults to None.
        title (str, optional): The title of the plot. Defaults to "".
        node_labels (Union[dict[int, str], dict[str, dict[int, str]], None], optional):
            Node labels for the plot.
            For homogeneous graphs, this is a dictionary mapping node IDs to labels.
            For heterogeneous graphs, this is a dictionary where keys are node types and values are dictionaries mapping node IDs to labels.
            Defaults to None.
        edge_weight_name (Union[str, None], optional): The name of the edge data field to use for edge weights.
            If provided, edge colors will be based on these weights. Defaults to None.
        ntype_colors (Union[list[Color], None], optional): A list of colors for node types. Defaults to None.
        etype_colors (Union[list[Color], None], optional): A list of colors for edge types. Defaults to None.
        figsize (tuple[int, int], optional): The figure size. Defaults to (6, 4).
        reverse_etypes (Union[dict[str, str], None], optional): A dictionary specifying reverse edge types.
            Used for representing bidirectional edges in heterogeneous graphs.
            Keys are edge types, and values are the corresponding reverse edge types. Defaults to None.
        **kwargs: Additional keyword arguments to pass to `nx.draw`.

    Returns:
        matplotlib.axes.Axes: The matplotlib axes object.

    Raises:
        ValueError: If `n_hop` is not greater than 0, if `fanouts` length is not equal to `n_hop`, or if the target_nodes are not proper type for a given graph type.
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
