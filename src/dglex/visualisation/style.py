import dgl
import torch
import networkx as nx
import seaborn as sns
import matplotlib
from typing import Optional, Dict, List, Tuple, Union, Any
from dataclasses import dataclass
from collections import defaultdict

# Type aliases for better readability
Color = Any
Legend = matplotlib.patches.Patch

@dataclass
class PlotConfig:
    """
    Configuration for graph visualization.

    Attributes:
        title (Optional[str]): Title of the plot.
        edge_weight_name (Optional[str]): Field name for edge weights in the DGL graph.
        node_palette (str): Seaborn palette name for nodes.
        edge_palette (str): Seaborn palette name for edges.
        figsize (Tuple[int, int]): Figure size (width, height).
        reverse_etypes (Optional[Dict[str, str]]): Mapping for reverse edge types (e.g., {"follow": "followed-by"}).
    """
    title: Optional[str] = None
    edge_weight_name: Optional[str] = None
    node_palette: str = "tab10"
    edge_palette: str = "tab10"
    figsize: Tuple[int, int] = (6, 4)
    reverse_etypes: Optional[Dict[str, str]] = None


def _get_colors(
    n_colors: int, 
    custom_colors: Optional[List[Color]] = None, 
    palette_name: str = "tab10"
) -> List[Color]:
    """
    Get a list of colors from a palette or custom list.
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
    """
    color = ntype_colors[0]
    legend = [matplotlib.patches.Patch(color=color, label="node")]
    return [color], legend


def _get_heterogenous_node_colors_and_legend(
    graph: dgl.DGLHeteroGraph,
    ng: nx.Graph,
    ntype_colors: List[Color],
) -> Tuple[List[Color], List[Legend]]:
    """
    Get colors and legend for a heterogeneous graph.
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


def _get_homogenous_node_labels(
    graph: dgl.DGLGraph, ng: nx.Graph, node_labels: Optional[Dict[int, str]] = None
) -> Dict[int, str]:
    """
    Generate node labels for a homogeneous graph.
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


def _get_heterogenous_node_labels(
    hetero_graph: dgl.DGLHeteroGraph,
    ng: nx.Graph,
    node_labels: Optional[Dict[str, Dict[int, str]]] = None,
) -> Dict[int, str]:
    """
    Generate node labels for a heterogeneous graph.
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


def _get_heterogeneous_edge_labels_reverse_etypes(
    hetero_graph: dgl.DGLHeteroGraph,
    ng: nx.Graph,
    edge_weight: Optional[Dict[Tuple[str, str, str], torch.Tensor]],
) -> Optional[Dict[Tuple[int, int], str]]:
    """
    Generate edge labels (weights) for a heterogeneous graph with reverse edge types.
    """
    return _get_heterogeneous_edge_labels(hetero_graph, ng, edge_weight)
