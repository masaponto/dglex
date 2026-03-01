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

# Save the preview as an image
dglex view my_graph.bin --output graph_preview.png
```

## LICENSE
MIT
