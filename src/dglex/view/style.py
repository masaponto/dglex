from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import dgl
import matplotlib
import networkx as nx
import seaborn as sns
import torch

from dglex.view.types import Color, Legend


def _get_colors(
    n_colors: int,
    custom_colors: Optional[List[Color]] = None,
    palette_name: str = "tab10"
) -> list:
    """
    Get a list of colors from a palette or custom list.

    Args:
        n_colors (int): Number of colors needed.
        custom_colors (Optional[List[Color]]): Pre-defined list of colors.
        palette_name (str): Name of the Seaborn palette to use if `custom_colors` is None.

    Returns:
        list: A list of `n_colors` colors.
    """
    if custom_colors is not None:
        if len(custom_colors) < n_colors:
            raise ValueError(f"Not enough custom colors. Expected {n_colors}, got {len(custom_colors)}")
        return custom_colors
    return sns.color_palette(palette_name, n_colors)


def _get_homogeneous_node_colors_and_legend(
    ntype_colors: List[Color],
) -> Tuple[List[Color], List[Legend]]:
    """
    Get colors and legend for a homogeneous graph.

    Args:
        ntype_colors (List[Color]): A list containing the color for the single node type.

    Returns:
        Tuple[List[Color], List[Legend]]: A tuple containing colors for each node and a legend element.
    """
    color = ntype_colors[0]
    legend = [matplotlib.patches.Patch(color=color, label="node")]
    return [color], legend


def _get_heterogeneous_node_colors_and_legend(
    graph: dgl.DGLHeteroGraph,
    ng: nx.Graph,
    ntype_colors: List[Color],
) -> Tuple[List[Color], List[Legend]]:
    """
    Get colors and legend for a heterogeneous graph.

    Args:
        graph (dgl.DGLHeteroGraph): The source DGL heterogeneous graph.
        ng (nx.Graph): The converted NetworkX graph.
        ntype_colors (List[Color]): A list of colors for each node type.

    Returns:
        Tuple[List[Color], List[Legend]]: A tuple containing colors for each node and legend elements per node type.
    """
    node_colors = []
    for node in ng.nodes(data=True):
        ntype_idx = node[1][dgl.NTYPE].item()
        node_colors.append(ntype_colors[ntype_idx])

    legend = [
        matplotlib.patches.Patch(color=ntype_colors[i], label=ntype)
        for i, ntype in enumerate(graph.ntypes)
    ]
    return node_colors, legend


def _get_homogeneous_edge_colors_and_legend(
    etype_colors: List[Color],
    edge_weight: Optional[torch.Tensor] = None,
) -> Tuple[List[Color], List[Legend]]:
    """
    Get colors and legend for a homogeneous edge.

    Args:
        etype_colors (List[Color]): A list containing the color for the single edge type.
        edge_weight (Optional[torch.Tensor]): Edge weight tensor. Currently unused for coloring.

    Returns:
        Tuple[List[Color], List[Legend]]: A tuple containing colors for each edge and a legend element.
    """
    color = etype_colors[0]
    legend = [matplotlib.patches.Patch(color=color, label="edge")]
    return [color], legend


def _get_heterogeneous_edge_colors_and_legend(
    graph: dgl.DGLHeteroGraph,
    ng: nx.Graph,
    etype_colors: List[Color],
    edge_weight: Optional[Dict[Tuple[str, str, str], torch.Tensor]] = None,
) -> Tuple[List[Color], List[Legend]]:
    """
    Get colors and legend for heterogeneous edges.

    Args:
        graph (dgl.DGLHeteroGraph): The source DGL heterogeneous graph.
        ng (nx.Graph): The converted NetworkX graph.
        etype_colors (List[Color]): Colors for each edge type.
        edge_weight (Optional[Dict]): Edge weight dictionary. Currently unused for coloring.

    Returns:
        Tuple[List[Color], List[Legend]]: A tuple containing colors for each edge and legend elements per edge type.
    """
    edge_colors = []
    for u, v, data in ng.edges(data=True):
        etype_idx = data[dgl.ETYPE].item()
        edge_colors.append(etype_colors[etype_idx])

    legend = [
        matplotlib.patches.Patch(color=etype_colors[i], label=str(etype))
        for i, etype in enumerate(graph.canonical_etypes)
    ]
    return edge_colors, legend


def _get_heterogeneous_edge_colors_and_legend_reverse_etypes(
    graph: dgl.DGLHeteroGraph,
    ng: nx.Graph,
    etype_colors: List[Color],
    ueid_master: Dict[int, Any],
    edge_weight: Optional[Dict[Tuple[str, str, str], torch.Tensor]] = None,
) -> Tuple[List[Color], List[Legend]]:
    """
    Get colors and legend for heterogeneous edges with reverse etypes.

    Args:
        graph (dgl.DGLHeteroGraph): The source DGL heterogeneous graph.
        ng (nx.Graph): The converted NetworkX graph.
        etype_colors (List[Color]): Colors for each unique edge identity (ueid).
        ueid_master (Dict[int, Any]): Mapping from etype ID to EdgeTypeInfo 相当オブジェクト。
            値は ``.ueid`` と ``.etype_legend_name`` 属性を持つことを想定する。
        edge_weight (Optional[Dict]): Edge weight dictionary. Currently unused for coloring.

    Returns:
        Tuple[List[Color], List[Legend]]: A tuple containing colors for each edge and legend elements for grouped edge types.
    """
    edge_colors = []
    for u, v, data in ng.edges(data=True):
        etype_idx = data[dgl.ETYPE].item()
        ueid = ueid_master[etype_idx].ueid
        edge_colors.append(etype_colors[ueid])

    # Unique legends based on ueid
    seen_ueids = set()
    legend = []
    for etype_idx in range(len(graph.etypes)):
        info = ueid_master[etype_idx]
        if info.ueid not in seen_ueids:
            legend.append(matplotlib.patches.Patch(color=etype_colors[info.ueid], label=info.etype_legend_name))
            seen_ueids.add(info.ueid)

    return edge_colors, legend


def _get_homogeneous_node_labels(
    graph: dgl.DGLGraph, ng: nx.Graph, node_labels: Optional[Dict[int, str]] = None
) -> Dict[int, str]:
    """
    Generate node labels for a homogeneous graph.

    Args:
        graph (dgl.DGLGraph): The source DGL graph.
        ng (nx.Graph): The converted NetworkX graph.
        node_labels (Optional[Dict[int, str]]): User-provided labels. If None, node IDs or dgl.NID are used.

    Returns:
        Dict[int, str]: Mapping from NetworkX node ID to label string.
    """
    if node_labels is None:
        if dgl.NID in graph.ndata:
            return {i: str(n) for i, n in enumerate(graph.ndata[dgl.NID].tolist())}
        else:
            return {i: str(i) for i in range(graph.num_nodes())}
    else:
        if dgl.NID in graph.ndata:
            # For subgraphs, graph.ndata[dgl.NID] contains original IDs.
            # We must handle the case where a node from the subgraph is not in the node_labels.
            orig_ids = graph.ndata[dgl.NID].tolist()
            return {
                i: node_labels.get(n, str(n)) for i, n in enumerate(orig_ids)
            }
        else:
            return {
                i: node_labels.get(i, str(i)) for i in range(graph.num_nodes())
            }


def _get_heterogeneous_node_labels(
    hetero_graph: dgl.DGLHeteroGraph,
    ng: nx.Graph,
    node_labels: Optional[Dict[str, Dict[int, str]]] = None,
) -> Dict[int, str]:
    """
    Generate node labels for a heterogeneous graph.

    Args:
        hetero_graph (dgl.DGLHeteroGraph): The source DGL heterogeneous graph.
        ng (nx.Graph): The converted NetworkX graph.
        node_labels (Optional[Dict[str, Dict[int, str]]]): User-provided labels per node type.

    Returns:
        Dict[int, str]: Mapping from NetworkX node ID to label string.
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


def _combine_edge_labels(
    src_nodes: List[int], dst_nodes: List[int], weights: List[float]
) -> Dict[Tuple[int, int], str]:
    """
    Combine weights for parallel edges into a single label string.

    Args:
        src_nodes (List[int]): List of source node IDs.
        dst_nodes (List[int]): List of destination node IDs.
        weights (List[float]): List of edge weights.

    Returns:
        Dict[Tuple[int, int], str]: Mapping from (src, dst) to combined weight string.
    """
    combined = defaultdict(list)
    for src, dst, weight in zip(src_nodes, dst_nodes, weights):
        combined[(src, dst)].append(f"{weight:.3f}")

    return {k: ", ".join(v) for k, v in combined.items()}


def _get_homogeneous_edge_labels(
    graph: dgl.DGLGraph, edge_weight: Optional[torch.Tensor]
) -> Optional[Dict[Tuple[int, int], str]]:
    """
    Generate edge labels (weights) for a homogeneous graph.

    Args:
        graph (dgl.DGLGraph): The source DGL graph.
        edge_weight (Optional[torch.Tensor]): Edge weight tensor.

    Returns:
        Optional[Dict[Tuple[int, int], str]]: Mapping from edge tuple to weight string, or None if no weight.
    """
    if edge_weight is None:
        return None

    src_nodes, dst_nodes = graph.edges()
    return _combine_edge_labels(
        src_nodes.tolist(), dst_nodes.tolist(), edge_weight.tolist()
    )


def _get_heterogeneous_edge_labels(
    hetero_graph: dgl.DGLHeteroGraph,
    ng: nx.Graph,
    edge_weight: Optional[Dict[Tuple[str, str, str], torch.Tensor]],
) -> Optional[Dict[Tuple[int, int], str]]:
    """
    Generate edge labels (weights) for a heterogeneous graph.

    Args:
        hetero_graph (dgl.DGLHeteroGraph): The source DGL heterogeneous graph.
        ng (nx.Graph): The converted NetworkX graph.
        edge_weight (Optional[Dict[Tuple[str, str, str], torch.Tensor]]): Dictionary of edge weight tensors per canonical edge type.

    Returns:
        Optional[Dict[Tuple[int, int], str]]: Mapping from edge tuple to weight string.
    """
    if edge_weight is None:
        return None

    # Pre-calculate weights for each etype to avoid redundant access
    etype_weights = {
        etype: edge_weight[etype].tolist()
        for etype in hetero_graph.canonical_etypes
    }

    combined = defaultdict(list)

    # Iterate through NetworkX edges and match with DGL weights
    for u, v, data in ng.edges(data=True):
        etype_idx = data[dgl.ETYPE].item()
        eid = data[dgl.EID].item()
        etype = hetero_graph.canonical_etypes[etype_idx]
        weight = etype_weights[etype][eid]
        combined[(u, v)].append(f"{weight:.3f}")

    return {k: ", ".join(v) for k, v in combined.items()}

