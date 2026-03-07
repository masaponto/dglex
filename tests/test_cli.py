import os
from unittest.mock import MagicMock, patch

import dgl
import pytest
import torch
import yaml

from dglex.cli import _decide_sampling_action, main

DEFAULT_LIMITS = {"max_nodes": 500, "max_edges": 2000}


@pytest.fixture
def temp_dgl_file(tmp_path):
    g = dgl.graph(([0, 1], [1, 2]))
    file_path = os.path.join(tmp_path, "test_graph.bin")
    dgl.save_graphs(file_path, [g])
    return file_path


def test_cli_view_basic(temp_dgl_file):
    with patch("matplotlib.pyplot.show"), \
         patch("dglex.visualisation.plot.plot_graph") as mock_plot:

        with patch("sys.argv", ["dglex", "view", temp_dgl_file]):
            main()

        mock_plot.assert_called_once()
        args, _ = mock_plot.call_args
        assert isinstance(args[0], (dgl.DGLGraph, dgl.DGLHeteroGraph))


def test_cli_view_save_output(temp_dgl_file, tmp_path):
    output_png = os.path.join(tmp_path, "output.png")
    with patch("matplotlib.pyplot.savefig") as mock_savefig, \
         patch("dglex.visualisation.plot.plot_graph"):

        with patch("sys.argv", ["dglex", "view", temp_dgl_file, "--output", output_png]):
            main()

        mock_savefig.assert_called_once_with(output_png)


def test_cli_view_with_options(temp_dgl_file):
    with patch("matplotlib.pyplot.show"), \
         patch("dglex.visualisation.plot.plot_graph") as mock_plot:

        reverse_etypes_json = '{"click": "clicked-by"}'
        with patch("sys.argv", ["dglex", "view", temp_dgl_file,
                                "--title", "Custom Title",
                                "--edge-weight", "w",
                                "--node-palette", "Set2",
                                "--edge-palette", "viridis",
                                "--figsize", "10", "8",
                                "--reverse-etypes", reverse_etypes_json]):
            main()

        _, kwargs = mock_plot.call_args
        assert kwargs["title"] == "Custom Title"
        assert kwargs["edge_weight_name"] == "w"
        assert kwargs["node_palette"] == "Set2"
        assert kwargs["edge_palette"] == "viridis"
        assert kwargs["figsize"] == (10, 8)
        assert kwargs["reverse_etypes"] == {"click": "clicked-by"}


def test_cli_view_invalid_palette(temp_dgl_file):
    with patch("dglex.visualisation.plot.plot_graph", side_effect=ValueError("is not a valid palette name")), \
         patch("sys.stderr", new_callable=MagicMock) as mock_stderr:

        with patch("sys.argv", ["dglex", "view", temp_dgl_file, "--node-palette", "invalid_palette"]):
            main()

        error_output = "".join(call.args[0] for call in mock_stderr.write.call_args_list)
        assert "Error: is not a valid palette name" in error_output


def test_cli_view_large_graph_sampling(tmp_path):
    # Create a "large" graph (over 500 nodes)
    src = torch.arange(600)
    dst = (src + 1) % 600
    g = dgl.graph((src, dst))
    file_path = os.path.join(tmp_path, "large_graph.bin")
    dgl.save_graphs(file_path, [g])

    with patch("matplotlib.pyplot.show"), \
         patch("dglex.visualisation.plot.plot_subgraph_with_neighbors") as mock_sub_plot, \
         patch("builtins.input", return_value="2"), \
         patch("sys.stdin.isatty", return_value=True):

        with patch("sys.argv", ["dglex", "view", file_path]):
            main()

        # Should call plot_subgraph_with_neighbors instead of plot_graph
        mock_sub_plot.assert_called_once()
        _, kwargs = mock_sub_plot.call_args
        assert "fanouts" in kwargs
        assert kwargs["n_hop"] == len(kwargs["fanouts"])


def test_cli_view_with_config_file(temp_dgl_file):
    config_data = {
        "view": {
            "node_palette": "magma",
            "reverse_etypes": {"follow": "followed-by"}
        }
    }
    config_path = os.path.join(os.getcwd(), "dglex.yaml")

    with open(config_path, "w") as f:
        yaml.dump(config_data, f)

    try:
        with patch("matplotlib.pyplot.show"), \
             patch("dglex.visualisation.plot.plot_graph") as mock_plot:

            with patch("sys.argv", ["dglex", "view", temp_dgl_file]):
                main()

            _, kwargs = mock_plot.call_args
            assert kwargs["node_palette"] == "magma"
            assert kwargs["reverse_etypes"] == {"follow": "followed-by"}
    finally:
        if os.path.exists(config_path):
            os.remove(config_path)


def test_decide_action_small_graph():
    assert _decide_sampling_action(100, 500, DEFAULT_LIMITS, False, True, False, "") == "continue"


def test_decide_action_force_override():
    assert _decide_sampling_action(600, 3000, DEFAULT_LIMITS, True, False, False, "") == "continue"


def test_decide_action_large_non_tty():
    assert _decide_sampling_action(600, 3000, DEFAULT_LIMITS, False, False, False, "") == "random"


def test_decide_action_large_with_output():
    assert _decide_sampling_action(600, 3000, DEFAULT_LIMITS, False, True, True, "") == "random"


def test_decide_action_user_continue():
    assert _decide_sampling_action(600, 3000, DEFAULT_LIMITS, False, True, False, "1") == "continue"


def test_decide_action_user_hub():
    assert _decide_sampling_action(600, 3000, DEFAULT_LIMITS, False, True, False, "3") == "hub"


def test_decide_action_user_abort():
    assert _decide_sampling_action(600, 3000, DEFAULT_LIMITS, False, True, False, "4") == "abort"


def test_decide_action_user_default_random():
    assert _decide_sampling_action(600, 3000, DEFAULT_LIMITS, False, True, False, "2") == "random"


@pytest.mark.parametrize("num_nodes,num_edges,expected", [
    (500, 2000, "continue"),  # 境界値: 以下はフル描画
    (501, 2000, "random"),    # 境界値: ノード超過
    (500, 2001, "random"),    # 境界値: エッジ超過
], ids=["at_limit", "node_over", "edge_over"])
def test_decide_action_size_boundary(num_nodes, num_edges, expected):
    action = _decide_sampling_action(
        num_nodes, num_edges, DEFAULT_LIMITS,
        force=False, is_tty=False, has_output=False, user_choice=""
    )
    assert action == expected


def test_cli_help():
    with patch("sys.stdout"), \
         patch("sys.argv", ["dglex", "--help"]):
        with pytest.raises(SystemExit) as e:
            main()
        assert e.value.code == 0
