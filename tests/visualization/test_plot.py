import pytest
import dgl
import numpy as np
import torch

# import matplotlib.pyplot as plt
from dglex.visualisation import plot_graph, plot_subgraph_with_neighbors

# from unittest.mock import patch
from pytest_mock import MockerFixture
import networkx


@pytest.fixture
def homogeneous_graph() -> dgl.DGLGraph:

    # randomly generated
    follow_src = np.array([6, 3, 7, 4, 6, 9, 2, 6, 7, 4])
    follow_dst = np.array([3, 7, 7, 2, 5, 4, 1, 7, 5, 1])

    homo_graph = dgl.graph((follow_src, follow_dst))

    homo_graph.edata["weight"] = torch.Tensor(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    )

    return homo_graph


@pytest.fixture
def heterogeneous_graph() -> dgl.DGLHeteroGraph:

    # randomly generated
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

    return hetero_graph


@pytest.fixture
def mock_nx_draw(mocker: MockerFixture):
    return mocker.patch("networkx.draw")


def test_plot_homogeneous_graph(
    homogeneous_graph: dgl.DGLGraph, mock_nx_draw: MockerFixture
):
    ax = plot_graph(homogeneous_graph)
    mock_nx_draw.assert_called_once()

    mock_nx_draw.reset_mock()  # reset mock for next test
    ax = plot_graph(homogeneous_graph, edge_wegihts="weight")

    mock_nx_draw.assert_called_once()


def test_plot_heterogeneous_graph(
    heterogeneous_graph: dgl.DGLHeteroGraph, mock_nx_draw: MockerFixture
):

    ax = plot_graph(heterogeneous_graph)
    mock_nx_draw.assert_called_once()
    mock_nx_draw.reset_mock()  # reset mock for next test

    # add revere etypes
    reverse_etypes = {
        "click": "clicked-by",
        "dislike": "disliked-by",
        "follow": "followed-by",
        "followed-by": "follow",
        "clicked-by": "click",
        "disliked-by": "dislike",
    }
    ax = plot_graph(
        heterogeneous_graph,
        figsize=(10, 10),
        title="graph",
        reverse_etypes=reverse_etypes,
    )

    mock_nx_draw.assert_called_once()
    mock_nx_draw.reset_mock()

    # test with edge weights
    ax = plot_graph(heterogeneous_graph, edge_wegihts="weight")
    mock_nx_draw.assert_called_once()
    mock_nx_draw.reset_mock()

    # test with edge weights and reverse etypes
    ax = plot_graph(
        heterogeneous_graph,
        edge_wegihts="weight",
        reverse_etypes=reverse_etypes,
    )
    mock_nx_draw.assert_called_once()


def test_plot_subgraph_with_neighbors_homogeneous_graph(
    homogeneous_graph: dgl.DGLHeteroGraph,
    mock_nx_draw: MockerFixture,
):

    ax = plot_subgraph_with_neighbors(homogeneous_graph, target_nodes=[0], n_hop=1)
    mock_nx_draw.assert_called_once()
    mock_nx_draw.reset_mock()

    ax = plot_subgraph_with_neighbors(homogeneous_graph, target_nodes=[0, 1], n_hop=1)
    mock_nx_draw.assert_called_once()
    mock_nx_draw.reset_mock()

    # test with edge weights
    ax = plot_subgraph_with_neighbors(
        homogeneous_graph, target_nodes=[0, 1], n_hop=1, edge_wegihts="weight"
    )
    mock_nx_draw.assert_called_once()


def test_plot_subgraph_with_neighbors_heterogeneous_graph(
    heterogeneous_graph: dgl.DGLHeteroGraph,
    mock_nx_draw: MockerFixture,
):

    ax = plot_subgraph_with_neighbors(
        heterogeneous_graph, target_nodes={"item": [0]}, n_hop=1
    )
    mock_nx_draw.assert_called_once()
    mock_nx_draw.reset_mock()

    ax = plot_subgraph_with_neighbors(
        heterogeneous_graph, target_nodes={"user": [0]}, n_hop=1, fanouts=[1]
    )

    mock_nx_draw.assert_called_once()
    mock_nx_draw.reset_mock()

    ax = plot_subgraph_with_neighbors(
        heterogeneous_graph,
        target_nodes={"user": [0]},
        n_hop=2,
        fanouts=[
            {
                "click": 1,
                "clicked-by": 1,
                "dislike": 1,
                "disliked-by": 1,
                "follow": 1,
                "followed-by": 1,
            },
            {
                "click": 0,
                "clicked-by": 0,
                "dislike": 1,
                "disliked-by": 1,
                "follow": 0,
                "followed-by": 0,
            },
        ],
    )

    mock_nx_draw.assert_called_once()
