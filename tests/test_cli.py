import os
import pytest
import torch
import dgl
import yaml
import json
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
        args, _ = mock_plot.call_args
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
    with patch("dglex.cli.plot_graph", side_effect=ValueError("is not a valid palette name")), \
         patch("sys.stderr", new_callable=MagicMock) as mock_stderr:
        
        with patch("sys.argv", ["dglex", "view", temp_dgl_file, "--node-palette", "invalid_palette"]):
            main()
            
        error_output = "".join(call.args[0] for call in mock_stderr.write.call_args_list)
        assert "Error: is not a valid palette name" in error_output


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
        with patch("dglex.cli.plt.show"), \
             patch("dglex.cli.plot_graph") as mock_plot:
            
            with patch("sys.argv", ["dglex", "view", temp_dgl_file]):
                main()
                
            _, kwargs = mock_plot.call_args
            assert kwargs["node_palette"] == "magma"
            assert kwargs["reverse_etypes"] == {"follow": "followed-by"}
    finally:
        if os.path.exists(config_path):
            os.remove(config_path)


def test_cli_help():
    with patch("sys.stdout") as mock_stdout, \
         patch("sys.argv", ["dglex", "--help"]):
        with pytest.raises(SystemExit) as e:
            main()
        assert e.value.code == 0
