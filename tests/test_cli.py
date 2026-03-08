import os
from unittest.mock import MagicMock, patch

import dgl
import pytest
import torch
import yaml

from dglex.cli import main
from dglex.view import _decide_sampling_action

DEFAULT_LIMITS = {"max_nodes": 500, "max_edges": 2000}


@pytest.fixture
def temp_dgl_file(tmp_path):
    g = dgl.graph(([0, 1], [1, 2]))
    file_path = os.path.join(tmp_path, "test_graph.bin")
    dgl.save_graphs(file_path, [g])
    return file_path


def test_cli_view_basic(temp_dgl_file):
    with patch("matplotlib.pyplot.show"), \
         patch("dglex.view.plot.plot_graph") as mock_plot:

        with patch("sys.argv", ["dglex", "view", temp_dgl_file]):
            main()

        mock_plot.assert_called_once()
        args, _ = mock_plot.call_args
        assert isinstance(args[0], (dgl.DGLGraph, dgl.DGLHeteroGraph))


def test_cli_view_save_output(temp_dgl_file, tmp_path):
    output_png = os.path.join(tmp_path, "output.png")
    with patch("matplotlib.pyplot.savefig") as mock_savefig, \
         patch("dglex.view.plot.plot_graph"):

        with patch("sys.argv", ["dglex", "view", temp_dgl_file, "--output", output_png]):
            main()

        mock_savefig.assert_called_once_with(output_png)


def test_cli_view_with_options(temp_dgl_file):
    with patch("matplotlib.pyplot.show"), \
         patch("dglex.view.plot.plot_graph") as mock_plot:

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
    with patch("dglex.view.plot.plot_graph", side_effect=ValueError("is not a valid palette name")), \
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
         patch("dglex.view.plot.plot_subgraph_with_neighbors") as mock_sub_plot, \
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
             patch("dglex.view.plot.plot_graph") as mock_plot:

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


# ---------------------------------------------------------------------------
# info コマンドのテスト
# ---------------------------------------------------------------------------

@pytest.fixture
def homo_graph_file(tmp_path):
    """ノード特徴量付きの homogeneous グラフファイルを作成する。"""
    g = dgl.graph(([0, 1, 2], [1, 2, 0]))
    g.ndata["feat"] = torch.zeros(3, 4, dtype=torch.float32)
    file_path = os.path.join(tmp_path, "homo_graph.bin")
    dgl.save_graphs(file_path, [g])
    return file_path


@pytest.fixture
def hetero_graph_file(tmp_path):
    """エッジ特徴量付きの heterogeneous グラフファイルを作成する。"""
    g = dgl.heterograph(
        {
            ("user", "follows", "user"): ([0, 1], [1, 2]),
            ("user", "rates", "item"): ([0, 0], [0, 1]),
        },
        num_nodes_dict={"user": 3, "item": 2},
    )
    g.nodes["user"].data["age"] = torch.ones(3, 1, dtype=torch.float32)
    g.edges[("user", "rates", "item")].data["weight"] = torch.ones(2, 1, dtype=torch.float32)
    file_path = os.path.join(tmp_path, "hetero_graph.bin")
    dgl.save_graphs(file_path, [g])
    return file_path


def test_info_homo_graph_summary(homo_graph_file, capsys):
    """homogeneous グラフの Graph Summary / Nodes / Edges が出力されることを確認。"""
    with patch("sys.argv", ["dglex", "info", homo_graph_file]):
        main()

    captured = capsys.readouterr()
    assert "Graph Summary" in captured.out
    assert "Graphs in file : 1" in captured.out
    assert "Graph type     : homogeneous" in captured.out
    assert "Nodes" in captured.out
    assert "Edges" in captured.out


def test_info_hetero_graph_ntypes_etypes(hetero_graph_file, capsys):
    """heterogeneous グラフの ntype / etype ごとの表示を確認。"""
    with patch("sys.argv", ["dglex", "info", hetero_graph_file]):
        main()

    captured = capsys.readouterr()
    assert "Graph type     : heterogeneous" in captured.out
    assert "user" in captured.out
    assert "item" in captured.out
    assert "user -> user" in captured.out or "follows" in captured.out
    assert "user -> item" in captured.out or "rates" in captured.out


def test_info_node_features(homo_graph_file, capsys):
    """Node Features セクションに dtype と shape が出力されることを確認。"""
    with patch("sys.argv", ["dglex", "info", homo_graph_file]):
        main()

    captured = capsys.readouterr()
    assert "Node Features" in captured.out
    assert "float32" in captured.out
    assert "(3, 4)" in captured.out


def test_info_edge_features_shown_when_present(hetero_graph_file, capsys):
    """エッジ特徴量が存在する場合に Edge Features セクションが出力されることを確認。"""
    with patch("sys.argv", ["dglex", "info", hetero_graph_file]):
        main()

    captured = capsys.readouterr()
    assert "Edge Features" in captured.out
    assert "weight" in captured.out


def test_info_no_edge_features_section_when_absent(homo_graph_file, capsys):
    """エッジ特徴量がない場合に Edge Features セクションが出力されないことを確認。"""
    with patch("sys.argv", ["dglex", "info", homo_graph_file]):
        main()

    captured = capsys.readouterr()
    assert "Edge Features" not in captured.out


def test_info_file_not_found(tmp_path, capsys):
    """存在しないファイルを指定した場合に 'Error: file not found' が stderr に出力されることを確認。"""
    missing_path = os.path.join(tmp_path, "nonexistent.bin")
    with patch("sys.argv", ["dglex", "info", missing_path]):
        main()

    captured = capsys.readouterr()
    assert "Error: file not found" in captured.err


def test_info_invalid_file(tmp_path, capsys):
    """無効なファイルを指定した場合に 'Error: file does not contain a valid DGL graph' が stderr に出力されることを確認。"""
    invalid_file = os.path.join(tmp_path, "invalid.bin")
    with open(invalid_file, "w") as f:
        f.write("not a dgl file")

    with patch("sys.argv", ["dglex", "info", invalid_file]):
        main()

    captured = capsys.readouterr()
    assert "Error: file does not contain a valid DGL graph" in captured.err
