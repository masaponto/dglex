from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class FeatureInfo:
    """テンソル特徴量のメタ情報。

    Attributes:
        dtype: テンソルのデータ型文字列（例: "float32"）。
        shape: テンソルの形状タプル（例: (3, 4)）。
    """

    dtype: str
    shape: Tuple[int, ...]


@dataclass
class DegreeStats:
    """次数統計情報。

    Attributes:
        mean: 平均次数。
        median: 中央値次数。
        min: 最小次数。
        max: 最大次数。
    """

    mean: float
    median: float
    min: float
    max: float


@dataclass
class NodeDegreeStats:
    """ノード種別ごとの in/out 次数統計。

    Attributes:
        in_degree: 入次数の統計情報。
        out_degree: 出次数の統計情報。
    """

    in_degree: DegreeStats
    out_degree: DegreeStats


@dataclass
class GraphInfo:
    """グラフ backend 非依存の構造情報をまとめた dataclass。

    Attributes:
        backend: グラフ backend 名。例: "dgl", "pyg"。
        graph_type: "homogeneous" または "heterogeneous"。
        graphs_count: ファイル内のグラフ総数。
        num_nodes: ノードタイプごとのノード数辞書。未定義の場合は `None`。
        num_edges: エッジタイプごとのエッジ数辞書。homogeneous の場合は {"_E": n}。
        node_features: ノード特徴量の辞書。キーは "ntype.feat_name"（homogeneous の場合は "feat_name"）。
        edge_features: エッジ特徴量の辞書。キーは "src->dst.feat_name"（homogeneous の場合は "feat_name"）。
        degree_stats: ノード種別ごとの in/out 次数統計。
        warnings: 表示時に伝える補足情報。
        summary: CLI と同じテキスト形式の要約文字列。
    """

    backend: str
    graph_type: str
    graphs_count: int
    num_nodes: Dict[str, int | None]
    num_edges: Dict[str, int]
    node_features: Dict[str, FeatureInfo]
    edge_features: Dict[str, FeatureInfo]
    summary: str
    degree_stats: Dict[str, NodeDegreeStats] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        """summary テキストを返す。

        Returns:
            str: CLI と同じ形式の要約テキスト。
        """
        return self.summary

    def __repr__(self) -> str:
        """Jupyter 等での評価時に summary テキストを返す。

        Returns:
            str: CLI と同じ形式の要約テキスト。
        """
        return self.summary
