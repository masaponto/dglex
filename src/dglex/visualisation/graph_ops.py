import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import dgl
import torch


@dataclass
class EdgeTypeInfo:
    """
    Information about an edge type, used for mapping original edge types to unique identities (ueid).

    Attributes:
        etype (str): Original edge type name.
        eid (int): Edge type ID in the DGL graph.
        ueid (int): Unique identity for grouped edge types (e.g., forward and reverse).
        etype_legend_name (str): Label to display in the legend.
    """
    etype: str
    eid: int
    ueid: int
    etype_legend_name: str

def get_random_nodes(
    graph: Union[dgl.DGLGraph, dgl.DGLHeteroGraph], k: int
) -> Union[List[int], Dict[str, List[int]]]:
    """
    Select k random nodes from the graph.

    Args:
        graph (Union[dgl.DGLGraph, dgl.DGLHeteroGraph]): The DGL graph.
        k (int): Number of nodes to select.

    Returns:
        Union[List[int], Dict[str, List[int]]]: Selected node IDs.
    """
    if graph.is_homogeneous:
        num_nodes = graph.num_nodes()
        if num_nodes <= k:
            return list(range(num_nodes))
        return torch.randperm(num_nodes)[:k].tolist()
    else:
        # Collect all possible (ntype, nid) pairs
        all_nodes = []
        for ntype in graph.ntypes:
            n_count = graph.num_nodes(ntype)
            for nid in range(n_count):
                all_nodes.append((ntype, nid))

        selected = random.sample(all_nodes, min(k, len(all_nodes)))

        result = defaultdict(list)
        for ntype, nid in selected:
            result[ntype].append(nid)
        return dict(result)

def get_top_k_degree_nodes(
    graph: Union[dgl.DGLGraph, dgl.DGLHeteroGraph], k: int
) -> Union[List[int], Dict[str, List[int]]]:
    """
    Select k nodes with the highest total degree (in + out).

    Args:
        graph (Union[dgl.DGLGraph, dgl.DGLHeteroGraph]): The DGL graph.
        k (int): Number of nodes to select.

    Returns:
        Union[List[int], Dict[str, List[int]]]: Selected node IDs.
    """
    if graph.is_homogeneous:
        degrees = graph.in_degrees() + graph.out_degrees()
        if len(degrees) == 0:
            return []
        _, indices = torch.topk(degrees, min(k, len(degrees)))
        return indices.tolist()
    else:
        total_degrees = {ntype: torch.zeros(graph.num_nodes(ntype)) for ntype in graph.ntypes}

        for srctype, etype, dsttype in graph.canonical_etypes:
            total_degrees[srctype] += graph.out_degrees(etype=etype)
            total_degrees[dsttype] += graph.in_degrees(etype=etype)

        all_nodes_with_degree = []
        for ntype in graph.ntypes:
            for nid, degree in enumerate(total_degrees[ntype].tolist()):
                all_nodes_with_degree.append((ntype, nid, degree))

        # Sort by degree descending
        all_nodes_with_degree.sort(key=lambda x: x[2], reverse=True)
        selected = all_nodes_with_degree[:min(k, len(all_nodes_with_degree))]

        result = defaultdict(list)
        for ntype, nid, _ in selected:
            result[ntype].append(nid)
        return dict(result)

def _count_heterogeneous_edges(
    graph: dgl.DGLHeteroGraph, reverse_etypes: Dict[str, str]
) -> int:
    """
    Count the number of unique edge identities in a heterogeneous graph, considering reverse types.

    Args:
        graph (dgl.DGLHeteroGraph): The DGL heterogeneous graph.
        reverse_etypes (Dict[str, str]): Mapping of edge types to their reverse counterparts.

    Returns:
        int: The number of unique edge identities.
    """
    num_directed_edge = len(
        [etype for etype in graph.etypes if etype not in reverse_etypes]
    )
    num_undirected_edge = len(reverse_etypes) // 2
    return num_directed_edge + num_undirected_edge

def _create_ueid_master(
    hetero_graph: dgl.DGLHeteroGraph, reverse_etypes: Dict[str, str]
) -> Dict[int, EdgeTypeInfo]:
    """
    Create a master mapping from original edge type IDs to EdgeTypeInfo.

    This groups reverse edge types together under a single unique identity (ueid).

    Args:
        hetero_graph (dgl.DGLHeteroGraph): The DGL heterogeneous graph.
        reverse_etypes (Dict[str, str]): Mapping of edge types to their reverse counterparts.

    Returns:
        Dict[int, EdgeTypeInfo]: Mapping from original edge type ID to its grouping info.
    """
    ueid_master = {}
    etype_eid_master = {etype: i for i, etype in enumerate(hetero_graph.etypes)}

    reverse_etype_tuples = []
    for k, v in reverse_etypes.items():
        if (v, k) not in reverse_etype_tuples:
            reverse_etype_tuples.append((k, v))

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

def _get_num_workers() -> int:
    """macOS では multiprocessing 非対応のため 0 を返す。

    Returns:
        int: macOS の場合は 0、それ以外は 1。
    """
    return 0 if sys.platform == "darwin" else 1


def _run_dataloader(dataloader: Any) -> tuple:
    """DataLoader から1バッチ取り出す（macOS/Linux 対応）。

    Args:
        dataloader: DGL の DataLoader インスタンス。

    Returns:
        tuple: (input_nodes, output_nodes, blocks) のタプル。
    """
    if sys.platform == "darwin":
        return next(iter(dataloader))
    with dataloader.enable_cpu_affinity():
        return next(iter(dataloader))


def _validate_edge_weight(edge_weight: Optional[torch.Tensor], shape: Any):
    """
    Validate that the edge weight tensor is 1D.

    Args:
        edge_weight (Optional[torch.Tensor]): The edge weight tensor to validate.
        shape (Any): The expected shape or current shape info for error reporting.

    Raises:
        ValueError: If `edge_weight` is not 1D.
    """
    if edge_weight is not None and edge_weight.ndimension() > 1:
        raise ValueError(f"edge_weight must be 1D. The current shape is {shape}")

def _get_edge_weight(graph: Union[dgl.DGLGraph, dgl.DGLHeteroGraph], edge_weight_name: Optional[str]) -> Any:
    """
    Retrieve edge weight data from a DGL graph.

    Args:
        graph (Union[dgl.DGLGraph, dgl.DGLHeteroGraph]): The DGL graph.
        edge_weight_name (Optional[str]): The name of the edge data field.

    Returns:
        Any: The edge weight data (Tensor or Dict of Tensors), or None.
    """
    return graph.edata[edge_weight_name] if edge_weight_name else None

def _extract_subgraph_nhop(
    g: dgl.DGLGraph,
    target_nodes: List[int],
    n_hop: int,
    node_labels: Optional[Dict[int, str]] = None,
) -> Tuple[dgl.DGLGraph, Dict[int, str]]:
    """
    Extract an n-hop subgraph for a homogeneous graph using full neighbor inclusion.

    Args:
        g (dgl.DGLGraph): The source homogeneous graph.
        target_nodes (List[int]): Starting node IDs.
        n_hop (int): Number of hops to expand.
        node_labels (Optional[Dict[int, str]]): Original node labels.

    Returns:
        Tuple[dgl.DGLGraph, Dict[int, str]]: The extracted subgraph and updated labels for the sub-nodes.
    """
    seed_nodes = torch.tensor(target_nodes)
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
    target_nodes: Dict[str, List[int]],
    n_hop: int,
    node_labels: Optional[Dict[str, Dict[int, str]]] = None,
) -> Tuple[dgl.DGLHeteroGraph, Dict[str, Dict[int, str]]]:
    """
    Extract an n-hop subgraph for a heterogeneous graph using full neighbor inclusion.

    Args:
        hetero_graph (dgl.DGLHeteroGraph): The source heterogeneous graph.
        target_nodes (Dict[str, List[int]]): Starting node IDs per type.
        n_hop (int): Number of hops to expand.
        node_labels (Optional[Dict[str, Dict[int, str]]]): Original node labels.

    Returns:
        Tuple[dgl.DGLHeteroGraph, Dict[str, Dict[int, str]]]: The extracted subgraph and updated labels.
    """
    seed_nodes = {ntype: torch.tensor(nodes) for ntype, nodes in target_nodes.items()}
    for i in range(n_hop):
        if i == n_hop - 1:
            sg = dgl.in_subgraph(hetero_graph, seed_nodes, relabel_nodes=True)
            break
        sg = dgl.in_subgraph(hetero_graph, seed_nodes)
        new_seed_nodes: Dict[str, List[int]] = defaultdict(list)
        for src_ntype, etype, dst_ntype in hetero_graph.canonical_etypes:
            src, dst = sg.edges(etype=etype)
            new_seed_nodes[src_ntype].extend(src.tolist())
            new_seed_nodes[dst_ntype].extend(dst.tolist())
            new_seed_nodes[src_ntype] = list(set(new_seed_nodes[src_ntype]))
            new_seed_nodes[dst_ntype] = list(set(new_seed_nodes[dst_ntype]))
        seed_nodes = {ntype: torch.tensor(nodes) for ntype, nodes in new_seed_nodes.items()}

    sampled_node_labels = {}
    for ntype in hetero_graph.ntypes:
        sampled_node_labels[ntype] = {
            i: f"{ntype}_{n}" if node_labels is None else node_labels[ntype][n]
            for i, n in enumerate(sg.ndata[dgl.NID][ntype].tolist())
        }
    return sg, sampled_node_labels

def _extract_sub_graph_nhop_fanouts(
    graph: dgl.DGLGraph,
    target_nodes: List[int],
    fanouts: List[int],
    node_labels: Optional[Dict[int, str]],
    edge_weight_name: Optional[str],
) -> Tuple[dgl.DGLGraph, Dict[int, str]]:
    """
    Extract a subgraph using neighbor sampling with fanouts (homogeneous).

    Args:
        graph (dgl.DGLGraph): The source graph.
        target_nodes (List[int]): Root nodes.
        fanouts (List[int]): Number of neighbors to sample at each hop.
        node_labels (Optional[Dict[int, str]]): Original node labels.
        edge_weight_name (Optional[str]): Edge weight field to preserve.

    Returns:
        Tuple[dgl.DGLGraph, Dict[int, str]]: Sampled subgraph and labels.
    """
    sampler = dgl.dataloading.NeighborSampler(fanouts)
    dataloader = dgl.dataloading.DataLoader(
        graph,
        torch.tensor(target_nodes),
        sampler,
        batch_size=len(target_nodes),
        shuffle=False,
        drop_last=False,
        num_workers=_get_num_workers(),
    )

    input_nodes, output_nodes, blocks = _run_dataloader(dataloader)

    _src, _dst, _edge_weight = [], [], []
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
        sampled_node_labels[i] = f"{nodes}" if nodes not in node_labels else node_labels[nodes]

    sg = dgl.graph((_src, _dst))
    if edge_weight_name is not None:
        sg.edata[edge_weight_name] = torch.tensor(_edge_weight)
    return sg, sampled_node_labels

def _extract_heterogeneous_sub_graph_nhop_fanouts(
    graph: dgl.DGLHeteroGraph,
    target_nodes: Dict[str, List[int]],
    fanouts: Union[List[int], List[Dict[str, int]]],
    node_labels: Optional[Dict[str, Dict[int, str]]],
    edge_weight_name: Optional[str],
) -> Tuple[dgl.DGLHeteroGraph, Dict[str, Dict[int, str]]]:
    """
    Extract a subgraph using neighbor sampling with fanouts (heterogeneous).

    Args:
        graph (dgl.DGLHeteroGraph): The source graph.
        target_nodes (Dict[str, List[int]]): Root nodes per type.
        fanouts (Union[List[int], List[Dict[str, int]]]): Fanouts per hop.
        node_labels (Optional[Dict[str, Dict[int, str]]]): Original labels.
        edge_weight_name (Optional[str]): Edge weight field to preserve.

    Returns:
        Tuple[dgl.DGLHeteroGraph, Dict[str, Dict[int, str]]]: Sampled heterogeneous subgraph and labels.
    """
    sampler = dgl.dataloading.NeighborSampler(fanouts)
    seed_nodes = {ntype: torch.tensor(nodes) for ntype, nodes in target_nodes.items()}
    dataloader = dgl.dataloading.DataLoader(
        graph,
        seed_nodes,
        sampler,
        batch_size=sum(len(v) for v in target_nodes.values()),
        shuffle=False,
        drop_last=False,
        num_workers=_get_num_workers(),
    )

    input_nodes, output_nodes, blocks = _run_dataloader(dataloader)

    graph_dic, edge_weight_dic = {}, {}
    sampled_node_labels: Dict[str, Dict[int, str]] = {ntype: {} for ntype in graph.ntypes}
    node_labels = ({ntype: {} for ntype in graph.ntypes} if node_labels is None else node_labels)

    for src_ntype, etype, dst_ntype in graph.canonical_etypes:
        _src, _dst, _edge_weight = [], [], []
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
            sampled_node_labels[ntype][i] = f"{nodes}" if nodes not in node_labels[ntype] else node_labels[ntype][nodes]

    sg = dgl.heterograph(graph_dic)
    if edge_weight_name is not None:
        sg.edata[edge_weight_name] = edge_weight_dic
    return sg, sampled_node_labels

def _validate_target_nodes(
    graph: Union[dgl.DGLGraph, dgl.DGLHeteroGraph],
    target_nodes: Union[List[int], Dict[str, List[int]]],
) -> None:
    """
    Validate that target_nodes type matches the graph type.

    Args:
        graph (Union[dgl.DGLGraph, dgl.DGLHeteroGraph]): The DGL graph.
        target_nodes (Union[List[int], Dict[str, List[int]]]): Nodes to validate.

    Raises:
        ValueError: If type mismatch occurs.
    """
    if graph.is_homogeneous:
        if not isinstance(target_nodes, list):
            raise ValueError(f"The target_nodes must be a list for homogeneous graph. Got {type(target_nodes)}")
    else:
        if not isinstance(target_nodes, dict):
            raise ValueError(f"The target_nodes must be a dictionary for heterogeneous graph. Got {type(target_nodes)}")
        if any(not isinstance(value, list) for value in target_nodes.values()):
            raise ValueError("The target_nodes must be a dictionary of lists.")

def _extract_subgraph(
    graph: Union[dgl.DGLGraph, dgl.DGLHeteroGraph],
    target_nodes: Union[List[int], Dict[str, List[int]]],
    n_hop: int,
    fanouts: Optional[Union[List[int], List[Dict[str, int]]]],
    node_labels: Optional[Union[Dict[int, str], Dict[str, Dict[int, str]]]],
    edge_weight_name: Optional[str],
) -> Tuple[Union[dgl.DGLGraph, dgl.DGLHeteroGraph], Dict[Any, Any]]:
    """
    Generic subgraph extraction dispatcher.

    Args:
        graph (Union[dgl.DGLGraph, dgl.DGLHeteroGraph]): The DGL graph.
        target_nodes (Union[List[int], Dict[str, List[int]]]): Seed nodes.
        n_hop (int): Number of hops.
        fanouts (Optional[Union[List[int], List[Dict[str, int]]]]): Sampling fanouts.
        node_labels (Optional[Union[Dict[int, str], Dict[str, Dict[int, str]]]]): Original labels.
        edge_weight_name (Optional[str]): Edge weight field.

    Returns:
        Tuple[Union[dgl.DGLGraph, dgl.DGLHeteroGraph], Dict[Any, Any]]: Extracted subgraph and its labels.
    """
    if graph.is_homogeneous:
        # Cast for type safety
        homo_target_nodes = target_nodes # type: ignore
        homo_node_labels = node_labels # type: ignore
        homo_fanouts = fanouts # type: ignore
        if fanouts is None:
            return _extract_subgraph_nhop(graph, homo_target_nodes, n_hop, homo_node_labels)
        else:
            return _extract_sub_graph_nhop_fanouts(graph, homo_target_nodes, homo_fanouts, homo_node_labels, edge_weight_name)
    else:
        # Cast for type safety
        hetero_target_nodes = target_nodes # type: ignore
        hetero_node_labels = node_labels # type: ignore
        hetero_fanouts = fanouts # type: ignore
        if fanouts is None:
            return _extract_heterogeneous_subgraph_nhop(graph, hetero_target_nodes, n_hop, hetero_node_labels)
        else:
            return _extract_heterogeneous_sub_graph_nhop_fanouts(graph, hetero_target_nodes, hetero_fanouts, hetero_node_labels, edge_weight_name)
