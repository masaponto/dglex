import dgl
import numpy as np
import pytest
import torch
from pytest_mock import MockerFixture

from dglex.view import plot_graph, plot_subgraph_with_neighbors


@pytest.fixture
def homogeneous_graph() -> dgl.DGLGraph:
    follow_src = np.array([6, 3, 7, 4, 6, 9, 2, 6, 7, 4])
    follow_dst = np.array([3, 7, 7, 2, 5, 4, 1, 7, 5, 1])
    homo_graph = dgl.graph((follow_src, follow_dst))
    homo_graph.edata["weight"] = torch.Tensor(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    )
    return homo_graph

@pytest.fixture
def heterogeneous_graph_and_reverse_etypes() -> (
    tuple[dgl.DGLHeteroGraph, dict[str, str]]
):
    follow_src = np.array([6, 3, 7, 4, 6, 9, 2, 6, 7, 4])
    follow_dst = np.array([3, 7, 7, 2, 5, 4, 1, 7, 5, 1])
    click_src = np.array([4, 0, 9, 5, 8, 0, 9, 2, 6, 3])
    click_dst = np.array([8, 2, 4, 2, 6, 4, 8, 6, 1, 3])
    dislike_src = np.array([8, 1, 9, 8, 9, 4, 1, 3, 6, 7])
    dislike_dst = np.array([2, 0, 3, 1, 7, 3, 1, 5, 5, 9])

    hetero_graph = dgl.heterograph(
        {
            ("user", "follow", "user"): (follow_src, follow_dst),
            ("user", "followed-by", "user"): (follow_dst, follow_src),
            ("user", "click", "item"): (click_src, click_dst),
            ("item", "clicked-by", "user"): (click_dst, click_src),
            ("user", "dislike", "item"): (dislike_src, dislike_dst),
            ("item", "disliked-by", "user"): (dislike_dst, dislike_src),
        }
    )

    hetero_graph.edata["weight"] = {
        ("user", "follow", "user"): torch.Tensor(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        ),
        ("user", "followed-by", "user"): torch.Tensor(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        ),
        ("user", "click", "item"): torch.Tensor(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        ),
        ("item", "clicked-by", "user"): torch.Tensor(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        ),
        ("user", "dislike", "item"): torch.Tensor(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        ),
        ("item", "disliked-by", "user"): torch.Tensor(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        ),
    }

    reverse_etypes = {
        "click": "clicked-by",
        "dislike": "disliked-by",
        "follow": "followed-by",
        "followed-by": "follow",
        "clicked-by": "click",
        "disliked-by": "dislike",
    }
    return hetero_graph, reverse_etypes

@pytest.fixture
def mock_nx_draw(mocker: MockerFixture):
    return mocker.patch("networkx.draw")

@pytest.fixture
def mock_nx_draw_edge_labels(mocker: MockerFixture):
    return mocker.patch("networkx.draw_networkx_edge_labels")

# --- Refactored tests for plot_graph ---

@pytest.mark.parametrize("use_weight, node_palette", [
    (False, "tab10"),
    (True, "viridis"),
], ids=["homo_default", "homo_with_weight_and_palette"])
def test_plot_homogeneous_graph_refactored(
    homogeneous_graph: dgl.DGLGraph, mock_nx_draw: MockerFixture, use_weight, node_palette
):
    edge_weight_name = "weight" if use_weight else None
    plot_graph(homogeneous_graph, edge_weight_name=edge_weight_name, node_palette=node_palette)
    mock_nx_draw.assert_called_once()

@pytest.mark.parametrize("use_weight, use_reverse, node_palette", [
    (False, False, "tab10"),
    (True, False, "magma"),
    (False, True, "Set2"),
    (True, True, "rocket"),
], ids=["hetero_default", "hetero_weight", "hetero_reverse", "hetero_weight_reverse"])
def test_plot_heterogeneous_graph_refactored(
    heterogeneous_graph_and_reverse_etypes: tuple[dgl.DGLHeteroGraph, dict[str, str]],
    mock_nx_draw: MockerFixture,
    use_weight, use_reverse, node_palette
):
    heterogeneous_graph, reverse_etypes = heterogeneous_graph_and_reverse_etypes
    edge_weight_name = "weight" if use_weight else None
    rev = reverse_etypes if use_reverse else None

    plot_graph(
        heterogeneous_graph,
        edge_weight_name=edge_weight_name,
        reverse_etypes=rev,
        node_palette=node_palette,
        title="Test Hetero"
    )
    mock_nx_draw.assert_called_once()

# --- Refactored tests for plot_subgraph_with_neighbors ---

@pytest.mark.parametrize("target_nodes, n_hop, use_fanouts", [
    ([2], 1, False),
    ([1, 2], 1, False),
    ([0, 1], 2, False),
    ([1, 2], 2, True),
], ids=["homo_sub_1hop_single", "homo_sub_1hop_multi", "homo_sub_2hop", "homo_sub_2hop_fanout"])
def test_plot_subgraph_homogeneous_refactored(
    homogeneous_graph: dgl.DGLGraph, mock_nx_draw: MockerFixture, target_nodes, n_hop, use_fanouts
):
    fanouts = [2] * n_hop if use_fanouts else None
    plot_subgraph_with_neighbors(
        homogeneous_graph,
        target_nodes=target_nodes,
        n_hop=n_hop,
        fanouts=fanouts,
        edge_weight_name="weight"
    )
    mock_nx_draw.assert_called_once()

@pytest.mark.parametrize("target_nodes, n_hop, use_fanouts, use_reverse", [
    ({"item": [0]}, 1, False, False),
    ({"user": [0, 1]}, 1, True, False),
    ({"user": [0]}, 2, True, True),
], ids=["hetero_sub_1hop", "hetero_sub_1hop_fanout", "hetero_sub_2hop_reverse"])
def test_plot_subgraph_heterogeneous_refactored(
    heterogeneous_graph_and_reverse_etypes: tuple[dgl.DGLHeteroGraph, dict[str, str]],
    mock_nx_draw: MockerFixture,
    target_nodes, n_hop, use_fanouts, use_reverse
):
    heterogeneous_graph, reverse_etypes = heterogeneous_graph_and_reverse_etypes
    rev = reverse_etypes if use_reverse else None

    # Define fanouts if needed
    fanouts = None
    if use_fanouts:
        if n_hop == 1:
            fanouts = [1]
        else:
            fanouts = [{"follow": 1, "click": 1, "dislike": 1, "followed-by": 1, "clicked-by": 1, "disliked-by": 1}] * n_hop

    plot_subgraph_with_neighbors(
        heterogeneous_graph,
        target_nodes=target_nodes,
        n_hop=n_hop,
        fanouts=fanouts,
        reverse_etypes=rev,
        edge_weight_name="weight"
    )
    mock_nx_draw.assert_called_once()


# --- Specialized tests for Parallel Edges Label Combination ---

def test_plot_parallel_edges_label_combination_homo(
    mock_nx_draw: MockerFixture,
    mock_nx_draw_edge_labels: MockerFixture
):
    """
    Verify that multiple edge weights between the same nodes are combined with a comma for homogeneous graphs.
    """
    # Create parallel edges: 0 -> 1 (weight 0.5) and 0 -> 1 (weight 0.8)
    u = torch.tensor([0, 0, 1])
    v = torch.tensor([1, 1, 2])
    g = dgl.graph((u, v))
    g.edata['w'] = torch.tensor([0.5, 0.8, 1.0])

    plot_graph(g, edge_weight_name='w')

    # Check that draw_networkx_edge_labels was called with combined labels
    mock_nx_draw_edge_labels.assert_called_once()
    _, kwargs = mock_nx_draw_edge_labels.call_args
    edge_labels = kwargs['edge_labels']

    # The weights for (0, 1) should be combined
    assert edge_labels[(0, 1)] == "0.500, 0.800"
    assert edge_labels[(1, 2)] == "1.000"


def test_plot_parallel_edges_label_combination_hetero(
    mock_nx_draw: MockerFixture,
    mock_nx_draw_edge_labels: MockerFixture
):
    """
    Verify that multiple edge weights between the same nodes are combined with a comma for heterogeneous graphs.
    """
    # Create parallel edges with different etypes:
    # user_0 -> user_1 via 'follow' (weight 0.5)
    # user_0 -> user_1 via 'follow' (weight 0.8) - another parallel edge in same type
    g = dgl.heterograph({
        ('user', 'follow', 'user'): ([0, 0], [1, 1]),
    })

    # DGL: when only ONE edge type exists, edata['w'] must be a tensor directly,
    # not a dictionary mapping etype to tensor.
    g.edata['w'] = torch.tensor([0.5, 0.8])

    plot_graph(g, edge_weight_name='w')

    mock_nx_draw_edge_labels.assert_called_once()
    _, kwargs = mock_nx_draw_edge_labels.call_args
    edge_labels = kwargs['edge_labels']

    # NetworkX node IDs in dgl.to_homogeneous start with 0 for 'user'
    assert edge_labels[(0, 1)] == "0.500, 0.800"
