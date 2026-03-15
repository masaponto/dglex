import dgl
import pytest
import torch

from dglex import GraphInfo, get_graph_info
from dglex.info.types import DegreeStats, FeatureInfo, NodeDegreeStats
from tests.pyg_stubs import (
    FakePygData,
    FakePygEdgeStore,
    FakePygHeteroData,
    FakePygNodeStore,
)


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
    assert info.backend == "dgl"
    assert info.graph_type == "homogeneous"
    assert info.warnings == []


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
    assert "_N" in info.degree_stats
    assert isinstance(info.degree_stats["_N"], NodeDegreeStats)


def test_homo_degree_stats(homo_graph):
    """homogeneous グラフで in/out 次数統計が算出されることを確認。"""
    info = get_graph_info(homo_graph)
    stats = info.degree_stats["_N"]
    assert isinstance(stats.in_degree, DegreeStats)
    assert stats.in_degree.mean == pytest.approx(1.0)
    assert stats.in_degree.median == pytest.approx(1.0)
    assert stats.in_degree.min == pytest.approx(1.0)
    assert stats.in_degree.max == pytest.approx(1.0)
    assert stats.out_degree.mean == pytest.approx(1.0)
    assert stats.out_degree.median == pytest.approx(1.0)
    assert stats.out_degree.min == pytest.approx(1.0)
    assert stats.out_degree.max == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# heterogeneous グラフ
# ---------------------------------------------------------------------------

def test_hetero_graph_type(hetero_graph):
    """graph_type が "heterogeneous" であることを確認。"""
    info = get_graph_info(hetero_graph)
    assert info.backend == "dgl"
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


def test_hetero_degree_stats(hetero_graph):
    """heterogeneous グラフで ntype ごとの in/out 次数統計が算出されることを確認。"""
    info = get_graph_info(hetero_graph)

    user = info.degree_stats["user"]
    assert user.in_degree.mean == pytest.approx(2 / 3)
    assert user.in_degree.median == pytest.approx(1.0)
    assert user.in_degree.min == pytest.approx(0.0)
    assert user.in_degree.max == pytest.approx(1.0)
    assert user.out_degree.mean == pytest.approx(4 / 3)
    assert user.out_degree.median == pytest.approx(1.0)
    assert user.out_degree.min == pytest.approx(0.0)
    assert user.out_degree.max == pytest.approx(3.0)

    item = info.degree_stats["item"]
    assert item.in_degree.mean == pytest.approx(1.0)
    assert item.in_degree.median == pytest.approx(1.0)
    assert item.in_degree.min == pytest.approx(1.0)
    assert item.in_degree.max == pytest.approx(1.0)
    assert item.out_degree.mean == pytest.approx(0.0)
    assert item.out_degree.median == pytest.approx(0.0)
    assert item.out_degree.min == pytest.approx(0.0)
    assert item.out_degree.max == pytest.approx(0.0)


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


def test_empty_ntype_degree_stats_are_zero():
    """ノード数 0 の ntype は次数統計が 0.0 で埋まることを確認。"""
    g = dgl.heterograph(
        {("user", "follows", "user"): ([0], [1])},
        num_nodes_dict={"user": 2, "ghost": 0},
    )

    info = get_graph_info(g)
    ghost = info.degree_stats["ghost"]
    assert ghost.in_degree.mean == pytest.approx(0.0)
    assert ghost.in_degree.median == pytest.approx(0.0)
    assert ghost.in_degree.min == pytest.approx(0.0)
    assert ghost.in_degree.max == pytest.approx(0.0)
    assert ghost.out_degree.mean == pytest.approx(0.0)
    assert ghost.out_degree.median == pytest.approx(0.0)
    assert ghost.out_degree.min == pytest.approx(0.0)
    assert ghost.out_degree.max == pytest.approx(0.0)


@pytest.fixture
def pyg_data_graph():
    """ノード特徴量・エッジ特徴量付きの PyG homogeneous グラフスタブ。"""
    return FakePygData(
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.int64),
        num_nodes=3,
        attributes={
            "x": torch.zeros(3, 4, dtype=torch.float32),
            "edge_attr": torch.ones(3, 1, dtype=torch.float64),
        },
    )


@pytest.fixture
def pyg_hetero_graph():
    """ノード特徴量・エッジ特徴量付きの PyG heterogeneous グラフスタブ。"""
    return FakePygHeteroData(
        node_stores={
            "user": FakePygNodeStore(
                num_nodes=3,
                attributes={"x": torch.ones(3, 2, dtype=torch.float32)},
            ),
            "item": FakePygNodeStore(
                num_nodes=2,
                attributes={"embedding": torch.ones(2, 3, dtype=torch.float32)},
            ),
        },
        edge_stores={
            ("user", "follows", "user"): FakePygEdgeStore(
                edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.int64),
            ),
            ("user", "rates", "item"): FakePygEdgeStore(
                edge_index=torch.tensor([[0, 0], [0, 1]], dtype=torch.int64),
                attributes={"edge_weight": torch.ones(2, 1, dtype=torch.float32)},
            ),
        },
    )


@pytest.fixture
def pyg_hetero_graph_missing_num_nodes():
    """一部 node type に explicit `num_nodes` がない PyG heterogeneous グラフスタブ。"""
    return FakePygHeteroData(
        node_stores={
            "user": FakePygNodeStore(
                num_nodes=3,
                attributes={"x": torch.ones(3, 2, dtype=torch.float32)},
            ),
            "item": FakePygNodeStore(
                attributes={"embedding": torch.ones(2, 3, dtype=torch.float32)},
            ),
        },
        edge_stores={
            ("user", "follows", "user"): FakePygEdgeStore(
                edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.int64),
            ),
            ("user", "rates", "item"): FakePygEdgeStore(
                edge_index=torch.tensor([[0, 0], [0, 1]], dtype=torch.int64),
                attributes={"edge_weight": torch.ones(2, 1, dtype=torch.float32)},
            ),
        },
    )


def test_pyg_data_graph_type(pyg_data_graph):
    """PyG `Data` 互換オブジェクトが homogeneous として扱われることを確認。"""
    info = get_graph_info(pyg_data_graph)
    assert info.backend == "pyg"
    assert info.graph_type == "homogeneous"


def test_pyg_data_features_and_degrees(pyg_data_graph):
    """PyG `Data` の特徴量と次数統計が取得できることを確認。"""
    info = get_graph_info(pyg_data_graph)
    assert info.num_nodes == {"_N": 3}
    assert info.num_edges == {"_E": 3}
    assert info.node_features["x"].dtype == "float32"
    assert info.node_features["x"].shape == (3, 4)
    assert info.edge_features["edge_attr"].dtype == "float64"
    assert info.edge_features["edge_attr"].shape == (3, 1)
    assert info.degree_stats["_N"].in_degree.mean == pytest.approx(1.0)
    assert info.degree_stats["_N"].out_degree.mean == pytest.approx(1.0)


def test_pyg_hetero_graph_type(pyg_hetero_graph):
    """PyG `HeteroData` 互換オブジェクトが heterogeneous として扱われることを確認。"""
    info = get_graph_info(pyg_hetero_graph)
    assert info.backend == "pyg"
    assert info.graph_type == "heterogeneous"


def test_pyg_hetero_features_and_counts(pyg_hetero_graph):
    """PyG `HeteroData` のノード数、エッジ数、特徴量が取得できることを確認。"""
    info = get_graph_info(pyg_hetero_graph)
    assert info.num_nodes == {"user": 3, "item": 2}
    assert info.num_edges == {"user->user": 2, "user->item": 2}
    assert info.node_features["user.x"].shape == (3, 2)
    assert info.node_features["item.embedding"].shape == (2, 3)
    assert info.edge_features["user->item.edge_weight"].shape == (2, 1)
    assert info.degree_stats["user"].out_degree.max == pytest.approx(3.0)
    assert info.degree_stats["item"].in_degree.mean == pytest.approx(1.0)


def test_pyg_hetero_num_nodes_not_defined_summary_and_warnings(pyg_hetero_graph_missing_num_nodes):
    """PyG heterogeneous で explicit `num_nodes` がない場合は not defined と warning を返す。"""
    info = get_graph_info(pyg_hetero_graph_missing_num_nodes)
    assert info.num_nodes == {"user": 3, "item": None}
    assert "item.embedding" in info.node_features
    assert info.node_features["item.embedding"].shape == (2, 3)
    assert info.degree_stats["item"].in_degree.mean == pytest.approx(1.0)
    assert "item : not defined" in info.summary
    assert any("item.num_nodes is not explicitly defined" in warning for warning in info.warnings)
