# dglex
![workflow](https://github.com/masaponto/dglex/actions/workflows/test.yml/badge.svg)
A DGL Extention library.

## About
This library is an extention of [DGL](https://www.dgl.ai/). It provides some useful functions for Graph Neural Network (GNN) projects.

## Support OS
- MacOS Apple Silicon
- Linux x86_64

## Installation
```bash
pip install torch # see https://pytorch.org/get-started/locally/
pip install dgl # see https://www.dgl.ai/pages/start.html
pip install git+https://github.com/masaponto/dglex.git
```

## Example
### Graph Visualization
 See examples/visualization.ipynb. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/masaponto/dglex/blob/main/examples/visualization.ipynb)

### CLI Preview Tool
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

## LICENSE
MIT
