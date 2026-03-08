# dglex
![workflow](https://github.com/masaponto/dglex/actions/workflows/test.yml/badge.svg)
A DGL Extension library.

## About
This library is an extension of [DGL](https://www.dgl.ai/). It provides some useful functions for Graph Neural Network (GNN) projects.

## Features
- Graph visualization for DGL graphs
- CLI preview tool for `.bin` graph files
- Support for heterogeneous graphs
- Automatic sampling for large graphs
- Configurable via `dglex.yaml`

## Support OS
- macOS (Apple Silicon)
- Linux (x86_64)

## Installation

Install PyTorch and DGL first (see their official installation guides):

- https://pytorch.org/get-started/locally/
- https://www.dgl.ai/pages/start.html

Then install `dglex`:

```bash
pip install git+https://github.com/masaponto/dglex.git
```

## Usage
### CLI

#### `info` — Inspect graph file structure
Display node counts, edge counts, degree statistics (mean/median/min/max per node type), and feature information of a DGL graph file (`.bin`).

```bash
dglex info my_graph.bin
```

Example output (heterogeneous graph):

```
Graph Summary
-------------
Graphs in file : 1
Graph type     : heterogeneous

Nodes
-----
user : 3
item : 3

Edges
-----
user -> user : 3
user -> item : 4
item -> user : 4

Degree Statistics
-----------------
user in  : mean=2.33 median=2.00 min=1.00 max=4.00
user out : mean=2.33 median=2.00 min=1.00 max=4.00
item in  : mean=2.67 median=3.00 min=2.00 max=3.00
item out : mean=2.67 median=3.00 min=2.00 max=3.00

Node Features
-------------
user.age       : float32 (3, 1)
user.embedding : float32 (3, 16)
item.price     : float32 (3, 1)
item.embedding : float32 (3, 16)

Edge Features
-------------
user->item.weight : float32 (4, 1)
```

---

#### `view` — Visualize a graph
You can preview a DGL graph file (`.bin`) from the command line.

```bash
# Preview the first graph in the file
dglex view my_graph.bin

# Preview with specific index, title, and edge weight
dglex view my_graph.bin --index 1 --title "Sample Hetero Graph" --edge-weight "weights"

# Use reverse edge types to represent undirected edges (JSON string)
dglex view my_graph.bin --reverse-etypes '{"click": "clicked-by", "follow": "followed-by"}'

# Customize color palettes
dglex view my_graph.bin --node-palette "Set2" --edge-palette "viridis"

# Save the preview as an image
dglex view my_graph.bin --output graph_preview.png
```

### Jupyter Notebook
 See examples/visualization.ipynb. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/masaponto/dglex/blob/main/examples/visualization.ipynb)


### Configuration File (`dglex.yaml`)
You can also use a `dglex.yaml` file in your current directory to set default options for the `view` command. Command line arguments will override these settings.

Example `dglex.yaml`:
```yaml
view:
  node_palette: "magma"
  edge_palette: "viridis"
  figsize: [10, 8]
  edge_weight: "weight"
  reverse_etypes:
    click: "clicked-by"
    follow: "followed-by"
```

### Large Graph Handling
When a graph exceeds certain limits (default: 500 nodes or 2,000 edges), `dglex` will warn you and offer sampling options to prevent system hangs.

#### Sampling Strategies
1. **Random Sample (Default)**: Picks `sample_size` nodes randomly. This is the fastest method and works even for extremely large graphs.
2. **Hub Sample**: Picks `sample_size` nodes with the highest total degrees. This provides a better view of the graph's core structure but may take a moment to calculate degrees on large graphs.

Both strategies use `plot_subgraph_with_neighbors` with physical edge limits (`fanouts`) to ensure the resulting preview is always safe to render.

#### Customizing Limits
You can customize these behaviors in your `dglex.yaml`:
```yaml
view:
  limits:
    max_nodes: 1000      # Show warning if nodes > 1000
    max_edges: 5000      # Show warning if edges > 5000
    sample_size: 10      # Pick 10 seed nodes for sampling
    sample_fanouts: [20, 10] # 2-hop sampling: 20 edges at 1st hop, 10 at 2nd
```

## Troubleshooting
If you're using WSL (Windows Subsystem for Linux) and `dglex view` doesn't display a window, or if you encounter the error:

`FigureCanvasAgg is non-interactive, and thus cannot be shown`

Matplotlib may be using a non-interactive backend (`Agg`). This happens when no GUI toolkit is available.

Install an interactive backend:

```bash
pip install PyQt5
```

Alternatively, you can always save the graph as an image using the `--output` option:
```bash
dglex view my_graph.bin --output graph.png
```

## LICENSE
MIT
