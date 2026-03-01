import dgl
import torch
import sys
from typing import Union, Any
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class EdgeTypeInfo:
    etype: str
    eid: int
    ueid: int
    etype_legend_name: str

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

def _validate_edge_weight(edge_weight, shape):
    if edge_weight is not None and edge_weight.ndimension() > 1:
        raise ValueError(f"edge_weight must be 1D. The current shape is {shape}")

def _get_edge_weight(graph, edge_weight_name):
    return graph.edata[edge_weight_name] if edge_weight_name else None

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
        num_workers=(0 if sys.platform == "darwin" else 1),
    )

    if sys.platform == "darwin":
        input_nodes, output_nodes, blocks = next(iter(dataloader))
    else:
        with dataloader.enable_cpu_affinity():
            input_nodes, output_nodes, blocks = next(iter(dataloader))

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
        num_workers=(0 if sys.platform == "darwin" else 1),
    )

    if sys.platform == "darwin":
        input_nodes, output_nodes, blocks = next(iter(dataloader))
    else:
        with dataloader.enable_cpu_affinity():
            input_nodes, output_nodes, blocks = next(iter(dataloader))

    graph_dic, edge_weight_dic = {}, {}
    sampled_node_labels = {ntype: {} for ntype in graph.ntypes}
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
    target_nodes: Union[list[int], dict[str, list[int]]],
) -> None:
    if graph.is_homogeneous:
        if not isinstance(target_nodes, list):
            raise ValueError(f"The target_nodes must be a list for homogeneous graph. The current target_nodes is {target_nodes}")
    else:
        if not isinstance(target_nodes, dict):
            raise ValueError(f"The target_nodes must be a dictionary for heterogeneous graph. The current target_nodes is {target_nodes}")
        if any(not isinstance(value, list) for value in target_nodes.values()):
            raise ValueError(f"The target_nodes must be a dictionary of lists. The current target_nodes is {target_nodes}")

def _extract_subgraph(
    graph: Union[dgl.DGLGraph, dgl.DGLHeteroGraph],
    target_nodes: Union[list[int], dict[str, list[int]]],
    n_hop: int,
    fanouts: Union[list[int], list[dict[str, int]], None],
    node_labels: Union[dict[int, str], dict[str, dict[int, str]], None],
    edge_weight_name: Union[str, None],
) -> tuple[Union[dgl.DGLGraph, dgl.DGLHeteroGraph], dict[Any, Any]]:
    if graph.is_homogeneous:
        if fanouts is None:
            return _extract_subgraph_nhop(graph, target_nodes, n_hop, node_labels)
        else:
            return _extract_sub_graph_nhop_fanouts(graph, target_nodes, fanouts, node_labels, edge_weight_name)
    else:
        if fanouts is None:
            return _extract_heterogeneous_subgraph_nhop(graph, target_nodes, n_hop, node_labels)
        else:
            return _extract_heterogeneous_sub_graph_nhop_fanouts(graph, target_nodes, fanouts, node_labels, edge_weight_name)
