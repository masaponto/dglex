import dgl
import pytest
import torch

from dglex import GraphInfo, get_graph_info
from dglex.info.types import FeatureInfo


@pytest.fixture
def homo_graph():
    """ノード特徴量・エッジ特徴量付きの homogeneous グラフ。"""
    g = dgl.graph(([0, 1, 2], [1, 2, 0]))
    g.ndata["feat"] = torch.zeros(3, 4, dtype=torch.float32)
    g.edata["weight"] = torch.ones(3, 1, dtype=torch.float64)
    return g


@pytest.fixture
def hetero_graph():
    """ノード特徴量・エッジ特徴量付きの heterogeneous グラフ。"""
    g = dgl.heterograph(
        {
            ("user", "follows", "user"): ([0, 1], [1, 2]),
            ("user", "rates", "item"): ([0, 0], [0, 1]),
        },
        num_nodes_dict={"user": 3, "item": 2},
    )
    g.nodes["user"].data["age"] = torch.ones(3, 1, dtype=torch.float32)
    g.edges[("user", "rates", "item")].data["weight"] = torch.ones(2, 1, dtype=torch.float32)
    return g


# ---------------------------------------------------------------------------
# homogeneous グラフ
# ---------------------------------------------------------------------------

def test_homo_graph_type(homo_graph):
    """graph_type が "homogeneous" であることを確認。"""
    info = get_graph_info(homo_graph)
    assert info.graph_type == "homogeneous"


def test_homo_graphs_count(homo_graph):
    """graphs_count がデフォルト 1 であることを確認。"""
    info = get_graph_info(homo_graph)
    assert info.graphs_count == 1


def test_homo_graphs_count_custom(homo_graph):
    """graphs_count に任意の値を渡せることを確認。"""
    info = get_graph_info(homo_graph, graphs_count=5)
    assert info.graphs_count == 5


def test_homo_num_nodes(homo_graph):
    """num_nodes が {"_N": 3} であることを確認。"""
    info = get_graph_info(homo_graph)
    assert info.num_nodes == {"_N": 3}


def test_homo_num_edges(homo_graph):
    """num_edges が {"_E": 3} であることを確認。"""
    info = get_graph_info(homo_graph)
    assert info.num_edges == {"_E": 3}


def test_homo_node_features(homo_graph):
    """node_features に "feat" が float32 / (3,4) で格納されることを確認。"""
    info = get_graph_info(homo_graph)
    assert "feat" in info.node_features
    feat = info.node_features["feat"]
    assert isinstance(feat, FeatureInfo)
    assert feat.dtype == "float32"
    assert feat.shape == (3, 4)


def test_homo_edge_features(homo_graph):
    """edge_features に "weight" が float64 / (3,1) で格納されることを確認。"""
    info = get_graph_info(homo_graph)
    assert "weight" in info.edge_features
    w = info.edge_features["weight"]
    assert w.dtype == "float64"
    assert w.shape == (3, 1)


def test_homo_summary_contains_key_lines(homo_graph):
    """summary に Graph Summary / homogeneous / Nodes / Edges が含まれることを確認。"""
    info = get_graph_info(homo_graph)
    assert "Graph Summary" in info.summary
    assert "homogeneous" in info.summary
    assert "Nodes" in info.summary
    assert "Edges" in info.summary


def test_homo_str_equals_summary(homo_graph):
    """str(info) が summary と一致することを確認。"""
    info = get_graph_info(homo_graph)
    assert str(info) == info.summary


def test_homo_repr_equals_summary(homo_graph):
    """repr(info) が summary と一致することを確認。"""
    info = get_graph_info(homo_graph)
    assert repr(info) == info.summary


def test_homo_returns_graph_info_type(homo_graph):
    """返り値が GraphInfo 型であることを確認。"""
    info = get_graph_info(homo_graph)
    assert isinstance(info, GraphInfo)


# ---------------------------------------------------------------------------
# heterogeneous グラフ
# ---------------------------------------------------------------------------

def test_hetero_graph_type(hetero_graph):
    """graph_type が "heterogeneous" であることを確認。"""
    info = get_graph_info(hetero_graph)
    assert info.graph_type == "heterogeneous"


def test_hetero_num_nodes(hetero_graph):
    """num_nodes に user=3, item=2 が格納されることを確認。"""
    info = get_graph_info(hetero_graph)
    assert info.num_nodes == {"user": 3, "item": 2}


def test_hetero_num_edges(hetero_graph):
    """num_edges に user->user=2, user->item=2 が格納されることを確認。"""
    info = get_graph_info(hetero_graph)
    assert info.num_edges == {"user->user": 2, "user->item": 2}


def test_hetero_node_features(hetero_graph):
    """node_features に "user.age" が float32 / (3,1) で格納されることを確認。"""
    info = get_graph_info(hetero_graph)
    assert "user.age" in info.node_features
    age = info.node_features["user.age"]
    assert age.dtype == "float32"
    assert age.shape == (3, 1)


def test_hetero_edge_features(hetero_graph):
    """edge_features に "user->item.weight" が float32 / (2,1) で格納されることを確認。"""
    info = get_graph_info(hetero_graph)
    assert "user->item.weight" in info.edge_features
    w = info.edge_features["user->item.weight"]
    assert w.dtype == "float32"
    assert w.shape == (2, 1)


def test_hetero_summary_contains_ntypes(hetero_graph):
    """summary に user / item が含まれることを確認。"""
    info = get_graph_info(hetero_graph)
    assert "user" in info.summary
    assert "item" in info.summary


def test_hetero_no_node_features_for_item(hetero_graph):
    """item ノードに特徴量がないため node_features に "item.*" キーがないことを確認。"""
    info = get_graph_info(hetero_graph)
    item_keys = [k for k in info.node_features if k.startswith("item.")]
    assert item_keys == []


# ---------------------------------------------------------------------------
# グラフ特徴量なし
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("edges,expected_nodes,expected_edges", [
    (([0, 1], [1, 2]), {"_N": 3}, {"_E": 2}),
    (([0], [1]), {"_N": 2}, {"_E": 1}),
], ids=["three_nodes", "two_nodes"])
def test_no_features_graph(edges, expected_nodes, expected_edges):
    """特徴量なしグラフで node_features / edge_features が空辞書であることを確認。"""
    g = dgl.graph(edges)
    info = get_graph_info(g)
    assert info.node_features == {}
    assert info.edge_features == {}
    assert info.num_nodes == expected_nodes
    assert info.num_edges == expected_edges
