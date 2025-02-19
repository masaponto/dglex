import pytest
import dgl
import numpy as np
import matplotlib.pyplot as plt
from dglex.visualisation.plot import plot_graph, plot_subgraph_with_neighbors
from unittest.mock import patch


@pytest.fixture
def homogeneous_graph() -> dgl.DGLGraph:
    n_users = 10
    n_follows = 10
    np.random.seed(42)
    follow_src = np.random.randint(0, n_users, n_follows)
    follow_dst = np.random.randint(0, n_users, n_follows)

    homo_graph = dgl.graph((follow_src, follow_dst))

    return homo_graph


@pytest.fixture
def heterogeneous_graph() -> dgl.DGLHeteroGraph:
    n_users = 10
    n_items = 10
    n_follows = 10
    n_clicks = 10
    n_dislikes = 10

    np.random.seed(42)
    follow_src = np.random.randint(0, n_users, n_follows)
    follow_dst = np.random.randint(0, n_users, n_follows)
    click_src = np.random.randint(0, n_users, n_clicks)
    click_dst = np.random.randint(0, n_items, n_clicks)
    dislike_src = np.random.randint(0, n_users, n_dislikes)
    dislike_dst = np.random.randint(0, n_items, n_dislikes)

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


def test_plot_homogeneous_graph(homogeneous_graph):
    with patch("matplotlib.pyplot.plot") as mock_plot:
        ax = plot_graph(homogeneous_graph)
        plt.plot()
