from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib

# Type aliases for better readability
Color = Any
Legend = matplotlib.patches.Patch

@dataclass
class PlotConfig:
    """
    Configuration for graph visualization.

    Attributes:
        title (str): Title of the plot.
        figsize (Tuple[int, int]): Figure size (width, height).
        node_palette (str): Seaborn palette name for nodes.
        edge_palette (str): Seaborn palette name for edges.
        node_labels (Optional): User-provided node labels.
        edge_weight_name (Optional[str]): Field name for edge weights in the DGL graph.
        ntype_colors (Optional[List[Color]]): Custom colors for node types.
        etype_colors (Optional[List[Color]]): Custom colors for edge types.
        reverse_etypes (Optional[Dict[str, str]]): Mapping for reverse edge types.
    """
    title: str = ""
    figsize: Tuple[int, int] = (6, 4)
    node_palette: str = "tab10"
    edge_palette: str = "tab10"
    node_labels: Optional[Union[Dict[int, str], Dict[str, Dict[int, str]]]] = None
    edge_weight_name: Optional[str] = None
    ntype_colors: Optional[List[Color]] = None
    etype_colors: Optional[List[Color]] = None
    reverse_etypes: Optional[Dict[str, str]] = None
