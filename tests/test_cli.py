import os
import pytest
import torch
import dgl
from unittest.mock import patch, MagicMock
from dglex.cli import main


@pytest.fixture
def temp_dgl_file(tmp_path):
    g = dgl.graph(([0, 1], [1, 2]))
    file_path = os.path.join(tmp_path, "test_graph.bin")
    dgl.save_graphs(file_path, [g])
    return file_path


def test_cli_view_basic(temp_dgl_file):
    with patch("dglex.cli.plt.show"), \
         patch("dglex.cli.plot_graph") as mock_plot:
        
        # Simulate command line arguments: dglex view path/to/graph.bin
        with patch("sys.argv", ["dglex", "view", temp_dgl_file]):
            main()
            
        mock_plot.assert_called_once()
        # The first argument to plot_graph should be a DGLGraph
        args, kwargs = mock_plot.call_args
        assert isinstance(args[0], (dgl.DGLGraph, dgl.DGLHeteroGraph))


def test_cli_view_save_output(temp_dgl_file, tmp_path):
    output_png = os.path.join(tmp_path, "output.png")
    with patch("dglex.cli.plt.savefig") as mock_savefig, \
         patch("dglex.cli.plot_graph"):
        
        with patch("sys.argv", ["dglex", "view", temp_dgl_file, "--output", output_png]):
            main()
            
        mock_savefig.assert_called_once_with(output_png)


def test_cli_view_with_options(temp_dgl_file):
    with patch("dglex.cli.plt.show"), \
         patch("dglex.cli.plot_graph") as mock_plot:
        
        with patch("sys.argv", ["dglex", "view", temp_dgl_file, "--title", "Custom Title", "--edge-weight", "w"]):
            main()
            
        _, kwargs = mock_plot.call_args
        assert kwargs["title"] == "Custom Title"
        assert kwargs["edge_weight_name"] == "w"


def test_cli_help():
    with patch("sys.stdout") as mock_stdout, \
         patch("sys.argv", ["dglex", "--help"]):
        with pytest.raises(SystemExit) as e:
            main()
        assert e.value.code == 0
