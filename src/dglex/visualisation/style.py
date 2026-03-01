import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import dgl
import networkx as nx
from typing import Union, Any, Optional

# Type aliases
Color = Union[str, tuple[float, ...]]
Legend = plt.Line2D


def _get_colors(
    n: int, colors: Optional[list[Color]], palette_name: str = "tab10"
) -> list[Color]:
    """
    Generate a list of colors for node or edge types.

    If `colors` is provided, it returns them directly. Otherwise, it samples `n` colors
    from the specified Seaborn palette.

    Args:
        n (int): Number of colors to generate.
        colors (Optional[list[Color]]): Pre-defined list of colors.
        palette_name (str): Name of the Seaborn palette to use if `colors` is None. Defaults to "tab10".

    Returns:
        list[Color]: A list of `n` colors.
    """
    if colors is None:
        colors = sns.color_palette(palette=palette_name, n_colors=n)
    return colors


def _get_colormap_from_base_color(base_color: Color) -> mcolors.LinearSegmentedColormap:
    """
    Create a linear colormap ranging from a lighter version of the base color to the base color itself.

    Args:
        base_color (Color): The base color to use for the colormap.

    Returns:
        mcolors.LinearSegmentedColormap: The generated colormap.
    """
    light_color = sns.blend_palette(["white", base_color], as_cmap=False)[1]
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "custom_cmap", [light_color, base_color]
    )
    return cmap


def _get_colors_from_weights(
    base_color: Color, weight_list: list[float]
) -> list[Color]:
    """
    Generate colors for edges based on their weights using a colormap derived from a base color.

    Args:
        base_color (Color): The base color for the edge type.
        weight_list (list[float]): A list of edge weights to map to colors.

    Returns:
        list[Color]: A list of colors corresponding to the input weights.
    """
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
    """
    Generate node colors and a legend for a homogeneous graph.

    Args:
        ntype_colors (list[Color]): A list containing the color for the single node type.

    Returns:
        tuple[list[Color], list[Legend]]: A tuple containing:
            - A list of colors for each node (all the same for homogeneous).
            - A list containing a single legend element for nodes.
    """
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
    """
    Generate node colors and a legend for a heterogeneous graph.

    Args:
        hetero_graph (dgl.DGLHeteroGraph): The source DGL heterogeneous graph.
        ng (nx.Graph): The converted NetworkX graph (homogeneous representation).
        ntype_colors (list[Color]): A list of colors corresponding to each node type in `hetero_graph.ntypes`.

    Returns:
        tuple[list[Color], list[Legend]]: A tuple containing:
            - A list of colors for each node in the NetworkX graph.
            - A list of legend elements, one for each node type.
    """
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
    """
    Generate edge colors and a legend for a homogeneous graph.

    Args:
        etype_colors (list[Color]): A list containing the color for the single edge type.
        edge_weight (Any, optional): Edge weights for color mapping. Defaults to None.

    Returns:
        tuple[list[Color], list[Legend]]: A tuple containing:
            - A list of colors for each edge.
            - A list containing a single legend element for edges.
    """
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
    """
    Generate edge colors and a legend for a heterogeneous graph (without reverse edge types).

    Args:
        hetero_graph (dgl.DGLHeteroGraph): The source DGL heterogeneous graph.
        ng (nx.Graph): The converted NetworkX graph.
        etype_colors (list[Color]): Colors for each edge type.
        edge_weight (Any, optional): Edge weights per canonical edge type. Defaults to None.

    Returns:
        tuple[list[Color], list[Legend]]: A tuple containing colors for each edge and legend elements for each edge type.
    """
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
    """
    Generate edge colors and a legend for a heterogeneous graph, grouping reverse edge types.

    Args:
        hetero_graph (dgl.DGLHeteroGraph): The source DGL heterogeneous graph.
        ng (nx.Graph): The converted NetworkX graph.
        etype_colors (list[Color]): Colors for each unique edge identity (ueid).
        ueid_master (dict[int, Any]): Mapping from etype ID to EdgeTypeInfo (containing ueid).
        edge_weight (Any, optional): Edge weights per canonical edge type. Defaults to None.

    Returns:
        tuple[list[Color], list[Legend]]: A tuple containing colors for each edge and legend elements for grouped edge types.
    """
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
    graph: dgl.DGLGraph, ng: nx.Graph, node_labels: Optional[dict[int, str]]
) -> dict[int, str]:
    """
    Generate node labels for a homogeneous graph.

    Args:
        graph (dgl.DGLGraph): The source DGL graph.
        ng (nx.Graph): The converted NetworkX graph.
        node_labels (Optional[dict[int, str]]): User-provided labels. If None, node IDs are used.

    Returns:
        dict[int, str]: Mapping from node ID to label string.
    """
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
    node_labels: Optional[dict[str, dict[int, str]]],
) -> dict[int, str]:
    """
    Generate node labels for a heterogeneous graph.

    Args:
        hetero_graph (dgl.DGLHeteroGraph): The source DGL heterogeneous graph.
        ng (nx.Graph): The converted NetworkX graph.
        node_labels (Optional[dict[str, dict[int, str]]]): User-provided labels per node type.

    Returns:
        dict[int, str]: Mapping from NetworkX node ID to label string.
    """
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
) -> Optional[dict[tuple[int, int], str]]:
    """
    Generate edge labels (weights) for a homogeneous graph.

    Args:
        graph (dgl.DGLGraph): The source DGL graph.
        edge_weight (Any): Edge weight tensor.

    Returns:
        Optional[dict[tuple[int, int], str]]: Mapping from edge tuple to weight string, or None if no weight.
    """
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
) -> Optional[dict[tuple[int, int], str]]:
    """
    Generate edge labels (weights) for a heterogeneous graph.

    Args:
        hetero_graph (dgl.DGLHeteroGraph): The source DGL heterogeneous graph.
        ng (nx.Graph): The converted NetworkX graph.
        edge_weight (Any): Dictionary of edge weight tensors per canonical edge type.

    Returns:
        Optional[dict[tuple[int, int], str]]: Mapping from edge tuple to weight string.
    """
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
) -> Optional[dict[tuple[int, int], str]]:
    """
    Generate edge labels (weights) for a heterogeneous graph with reverse edge types.

    Args:
        hetero_graph (dgl.DGLHeteroGraph): The source DGL heterogeneous graph.
        ng (nx.Graph): The converted NetworkX graph.
        edge_weight (Any): Dictionary of edge weight tensors per canonical edge type.

    Returns:
        Optional[dict[tuple[int, int], str]]: Mapping from edge tuple to weight string.
    """
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
