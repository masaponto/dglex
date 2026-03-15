import statistics
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from dglex.info.types import DegreeStats, FeatureInfo, GraphInfo, NodeDegreeStats

if TYPE_CHECKING:
    import dgl
    import torch


class UnsupportedGraphError(ValueError):
    """サポート対象外のグラフオブジェクトを受け取ったことを示す例外。"""


@dataclass(frozen=True)
class NodeCountResolution:
    """表示用と内部処理用の node count 解決結果。

    Attributes:
        explicit: ユーザーが明示設定した node count。未設定なら `None`。
        inferred: 内部処理用に推定した node count。推定不能なら `None`。
        source: 推定に使った情報源。
    """

    explicit: int | None
    inferred: int | None
    source: str | None


class GraphInfoBackendAdapter(ABC):
    """backend ごとの差分を吸収して GraphInfo を構築する抽象アダプタ。"""

    def __init__(self, graph: object, graphs_count: int = 1) -> None:
        """アダプタを初期化する。

        Args:
            graph: backend 固有のグラフオブジェクト。
            graphs_count: ファイル内のグラフ総数。
        """
        self._graph = graph
        self._graphs_count = graphs_count

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """backend 名を返す。"""

    @property
    @abstractmethod
    def graph_type(self) -> str:
        """グラフ種別を返す。"""

    @abstractmethod
    def get_num_nodes(self) -> dict[str, int | None]:
        """ノード数辞書を返す。"""

    @abstractmethod
    def get_num_edges(self) -> dict[str, int]:
        """エッジ数辞書を返す。"""

    @abstractmethod
    def get_node_features(self) -> dict[str, FeatureInfo]:
        """ノード特徴量辞書を返す。"""

    @abstractmethod
    def get_edge_features(self) -> dict[str, FeatureInfo]:
        """エッジ特徴量辞書を返す。"""

    @abstractmethod
    def get_degree_stats(self) -> dict[str, NodeDegreeStats]:
        """次数統計辞書を返す。"""

    def get_warnings(self) -> list[str]:
        """表示向け warning 一覧を返す。

        Returns:
            list[str]: warning メッセージ一覧。
        """
        return []

    def to_graph_info(self) -> GraphInfo:
        """共通 GraphInfo へ変換する。

        Returns:
            GraphInfo: backend 非依存のグラフ情報。
        """
        return GraphInfo(
            backend=self.backend_name,
            graph_type=self.graph_type,
            graphs_count=self._graphs_count,
            num_nodes=self.get_num_nodes(),
            num_edges=self.get_num_edges(),
            node_features=self.get_node_features(),
            edge_features=self.get_edge_features(),
            degree_stats=self.get_degree_stats(),
            warnings=self.get_warnings(),
            summary="",
        )


def _build_degree_stats(values: list[int]) -> DegreeStats:
    """次数配列から統計情報を算出する。

    Args:
        values: 次数の整数配列。

    Returns:
        DegreeStats: 算出済みの統計情報。
    """
    if not values:
        return DegreeStats(mean=0.0, median=0.0, min=0.0, max=0.0)

    mean = float(sum(values)) / len(values)
    return DegreeStats(
        mean=mean,
        median=float(statistics.median(values)),
        min=float(min(values)),
        max=float(max(values)),
    )


def _feature_info_from_tensor(tensor: "torch.Tensor") -> FeatureInfo:
    """テンソルから特徴量メタ情報を構築する。

    Args:
        tensor: 対象テンソル。

    Returns:
        FeatureInfo: dtype と shape を持つ特徴量情報。
    """
    dtype = str(tensor.dtype).replace("torch.", "")
    return FeatureInfo(dtype=dtype, shape=tuple(tensor.shape))


def _is_torch_tensor(value: object) -> bool:
    """与えられた値が torch.Tensor かを返す。

    Args:
        value: 判定対象。

    Returns:
        bool: Tensor であれば True。
    """
    try:
        import torch
    except ModuleNotFoundError:
        return False

    return isinstance(value, torch.Tensor)


def _to_int_list(tensor: "torch.Tensor") -> list[int]:
    """テンソルを整数リストへ変換する。

    Args:
        tensor: 変換対象テンソル。

    Returns:
        list[int]: Python の整数配列。
    """
    return [int(value) for value in tensor.tolist()]


def _iter_store_items(store: object) -> Iterable[tuple[str, object]]:
    """PyG 風 store の属性列挙を返す。

    Args:
        store: `items()` を持つ store オブジェクト。

    Returns:
        Iterable[tuple[str, object]]: 属性名と値のペア列。

    Raises:
        UnsupportedGraphError: 列挙 API が見つからない場合。
    """
    items = getattr(store, "items", None)
    if callable(items):
        return list(items())

    raise UnsupportedGraphError("unsupported graph store: missing items()")


def _get_explicit_store_value(store: object, key: str) -> object | None:
    """store から明示設定済みの値を副作用なく取得する。

    Args:
        store: 対象 store。
        key: 取得するキー名。

    Returns:
        object | None: 明示設定されていればその値。
    """
    getter = getattr(store, "get", None)
    if callable(getter):
        return getter(key, None)

    try:
        for item_key, value in _iter_store_items(store):
            if item_key == key:
                return value
    except UnsupportedGraphError:
        pass

    store_vars = vars(store)
    if key in store_vars:
        return store_vars[key]

    return None


def _get_explicit_num_nodes(store: object) -> int | None:
    """明示設定された `num_nodes` を返す。

    Args:
        store: PyG graph または store オブジェクト。

    Returns:
        int | None: 明示設定済みノード数。
    """
    value = _get_explicit_store_value(store, "num_nodes")
    if isinstance(value, int):
        return value
    return None


def _infer_pyg_homo_num_nodes(graph: object) -> NodeCountResolution:
    """PyG homogeneous グラフの node count を解決する。

    Args:
        graph: PyG `Data` 互換オブジェクト。

    Returns:
        NodeCountResolution: node count の解決結果。
    """
    explicit = _get_explicit_num_nodes(graph)
    if explicit is not None:
        return NodeCountResolution(explicit=explicit, inferred=explicit, source="num_nodes")

    x = getattr(graph, "x", None)
    if _is_torch_tensor(x) and x.dim() > 0:
        return NodeCountResolution(explicit=None, inferred=int(x.size(0)), source="x")

    num_edges = _get_pyg_edge_count(getattr(graph, "edge_index", None))
    for name, value in _iter_store_items(graph):
        if not _is_torch_tensor(value):
            continue
        if value.dim() == 0 or name in {"edge_index", "adj_t", "face", "ptr"}:
            continue
        leading_dim = int(value.size(0))
        if name in {"edge_attr", "edge_weight"} and leading_dim == num_edges:
            continue
        return NodeCountResolution(explicit=None, inferred=leading_dim, source=name)

    edge_index = getattr(graph, "edge_index", None)
    if _is_torch_tensor(edge_index) and edge_index.numel() > 0:
        return NodeCountResolution(
            explicit=None,
            inferred=int(edge_index.max().item()) + 1,
            source="edge_index",
        )

    return NodeCountResolution(explicit=None, inferred=None, source=None)


def _get_pyg_edge_count(edge_index: object) -> int:
    """PyG の `edge_index` からエッジ数を返す。

    Args:
        edge_index: `edge_index` テンソル想定の値。

    Returns:
        int: エッジ数。
    """
    if not _is_torch_tensor(edge_index) or edge_index.dim() != 2:
        return 0

    return int(edge_index.size(1))


def _classify_pyg_homo_feature(
    name: str,
    value: object,
    num_nodes: int | None,
    num_edges: int,
) -> str | None:
    """PyG homogeneous 属性の feature 種別を判定する。

    Args:
        name: 属性名。
        value: 属性値。
        num_nodes: ノード数。
        num_edges: エッジ数。

    Returns:
        str | None: `"node"`, `"edge"`, `None` のいずれか。
    """
    if not _is_torch_tensor(value):
        return None

    tensor = value
    if name in {"edge_index", "adj_t", "face", "ptr"}:
        return None

    if tensor.dim() == 0:
        return None

    leading_dim = int(tensor.size(0))
    if name in {"edge_attr", "edge_weight"} and leading_dim == num_edges:
        return "edge"

    if num_nodes is not None and name in {"x", "pos", "batch"} and leading_dim == num_nodes:
        return "node"

    if num_nodes is not None and leading_dim == num_nodes and leading_dim != num_edges:
        return "node"

    if leading_dim == num_edges and leading_dim != num_nodes:
        return "edge"

    if num_nodes is not None and leading_dim == num_nodes == num_edges:
        return "edge" if name.startswith("edge_") else "node"

    if num_nodes is None and name.startswith("edge_") and leading_dim == num_edges:
        return "edge"

    if num_nodes is None and name not in {"edge_attr", "edge_weight"}:
        return "node"

    return None


def _classify_pyg_hetero_feature(name: str, value: object, target_size: int | None) -> bool:
    """PyG heterogeneous store の feature 採用可否を返す。

    Args:
        name: 属性名。
        value: 属性値。
        target_size: 先頭次元が一致すべきサイズ。

    Returns:
        bool: feature として採用するなら True。
    """
    if not _is_torch_tensor(value):
        return False

    tensor = value
    if name in {"edge_index", "adj_t", "ptr"} or tensor.dim() == 0:
        return False

    if target_size is None:
        return True

    return int(tensor.size(0)) == target_size


def _compute_pyg_degree_stats(edge_index: object, num_nodes: int) -> NodeDegreeStats:
    """PyG homogeneous グラフの次数統計を算出する。

    Args:
        edge_index: `edge_index` テンソル。
        num_nodes: ノード数。

    Returns:
        NodeDegreeStats: in/out 次数統計。
    """
    import torch

    if not _is_torch_tensor(edge_index) or edge_index.numel() == 0:
        zeros = [0] * num_nodes
        return NodeDegreeStats(
            in_degree=_build_degree_stats(zeros),
            out_degree=_build_degree_stats(zeros),
        )

    out_values = _to_int_list(torch.bincount(edge_index[0], minlength=num_nodes))
    in_values = _to_int_list(torch.bincount(edge_index[1], minlength=num_nodes))
    return NodeDegreeStats(
        in_degree=_build_degree_stats(in_values),
        out_degree=_build_degree_stats(out_values),
    )


class DglInfoAdapter(GraphInfoBackendAdapter):
    """DGL グラフ向けの info アダプタ。"""

    @property
    def backend_name(self) -> str:
        """backend 名を返す。

        Returns:
            str: `"dgl"`。
        """
        return "dgl"

    @property
    def graph_type(self) -> str:
        """グラフ種別を返す。

        Returns:
            str: homogeneous または heterogeneous。
        """
        graph = self._graph
        return "homogeneous" if graph.is_homogeneous else "heterogeneous"

    def get_num_nodes(self) -> dict[str, int]:
        """ノード数辞書を返す。

        Returns:
            dict[str, int]: ノード数辞書。
        """
        graph = self._graph
        if graph.is_homogeneous:
            return {"_N": int(graph.num_nodes())}

        return {ntype: int(graph.num_nodes(ntype)) for ntype in graph.ntypes}

    def get_num_edges(self) -> dict[str, int]:
        """エッジ数辞書を返す。

        Returns:
            dict[str, int]: エッジ数辞書。
        """
        graph = self._graph
        if graph.is_homogeneous:
            return {"_E": int(graph.num_edges())}

        return {
            f"{src_type}->{dst_type}": int(graph.num_edges((src_type, etype, dst_type)))
            for src_type, etype, dst_type in graph.canonical_etypes
        }

    def get_node_features(self) -> dict[str, FeatureInfo]:
        """ノード特徴量辞書を返す。

        Returns:
            dict[str, FeatureInfo]: ノード特徴量辞書。
        """
        graph = self._graph
        features: dict[str, FeatureInfo] = {}
        if graph.is_homogeneous:
            for feat_name, tensor in graph.ndata.items():
                features[feat_name] = _feature_info_from_tensor(tensor)
            return features

        for ntype in graph.ntypes:
            for feat_name, tensor in graph.nodes[ntype].data.items():
                features[f"{ntype}.{feat_name}"] = _feature_info_from_tensor(tensor)
        return features

    def get_edge_features(self) -> dict[str, FeatureInfo]:
        """エッジ特徴量辞書を返す。

        Returns:
            dict[str, FeatureInfo]: エッジ特徴量辞書。
        """
        graph = self._graph
        features: dict[str, FeatureInfo] = {}
        if graph.is_homogeneous:
            for feat_name, tensor in graph.edata.items():
                features[feat_name] = _feature_info_from_tensor(tensor)
            return features

        for src_type, etype, dst_type in graph.canonical_etypes:
            for feat_name, tensor in graph.edges[(src_type, etype, dst_type)].data.items():
                features[f"{src_type}->{dst_type}.{feat_name}"] = _feature_info_from_tensor(tensor)
        return features

    def get_degree_stats(self) -> dict[str, NodeDegreeStats]:
        """次数統計辞書を返す。

        Returns:
            dict[str, NodeDegreeStats]: ノード種別ごとの次数統計。
        """
        graph = self._graph
        degree_stats: dict[str, NodeDegreeStats] = {}

        if graph.is_homogeneous:
            degree_stats["_N"] = NodeDegreeStats(
                in_degree=_build_degree_stats([int(value) for value in graph.in_degrees().tolist()]),
                out_degree=_build_degree_stats([int(value) for value in graph.out_degrees().tolist()]),
            )
            return degree_stats

        for ntype in graph.ntypes:
            num_nodes = int(graph.num_nodes(ntype))
            in_degree_values = [0] * num_nodes
            out_degree_values = [0] * num_nodes

            for src_type, etype, dst_type in graph.canonical_etypes:
                if dst_type == ntype:
                    for idx, degree in enumerate(
                        graph.in_degrees(etype=(src_type, etype, dst_type)).tolist()
                    ):
                        in_degree_values[idx] += int(degree)
                if src_type == ntype:
                    for idx, degree in enumerate(
                        graph.out_degrees(etype=(src_type, etype, dst_type)).tolist()
                    ):
                        out_degree_values[idx] += int(degree)

            degree_stats[ntype] = NodeDegreeStats(
                in_degree=_build_degree_stats(in_degree_values),
                out_degree=_build_degree_stats(out_degree_values),
            )

        return degree_stats


class PygDataInfoAdapter(GraphInfoBackendAdapter):
    """PyG homogeneous グラフ向けの info アダプタ。"""

    @property
    def backend_name(self) -> str:
        """backend 名を返す。

        Returns:
            str: `"pyg"`。
        """
        return "pyg"

    @property
    def graph_type(self) -> str:
        """グラフ種別を返す。

        Returns:
            str: `"homogeneous"`。
        """
        return "homogeneous"

    def _resolve_num_nodes(self) -> NodeCountResolution:
        """homogeneous グラフの node count 解決結果を返す。

        Returns:
            NodeCountResolution: node count の解決結果。
        """
        return _infer_pyg_homo_num_nodes(self._graph)

    def get_num_nodes(self) -> dict[str, int | None]:
        """ノード数辞書を返す。

        Returns:
            dict[str, int]: homogeneous 用ノード数辞書。
        """
        resolution = self._resolve_num_nodes()
        return {"_N": resolution.explicit}

    def get_num_edges(self) -> dict[str, int]:
        """エッジ数辞書を返す。

        Returns:
            dict[str, int]: homogeneous 用エッジ数辞書。
        """
        return {"_E": _get_pyg_edge_count(getattr(self._graph, "edge_index", None))}

    def get_node_features(self) -> dict[str, FeatureInfo]:
        """ノード特徴量辞書を返す。

        Returns:
            dict[str, FeatureInfo]: ノード特徴量辞書。
        """
        num_nodes = self._resolve_num_nodes().inferred
        num_edges = _get_pyg_edge_count(getattr(self._graph, "edge_index", None))
        features: dict[str, FeatureInfo] = {}
        for feat_name, value in _iter_store_items(self._graph):
            if _classify_pyg_homo_feature(feat_name, value, num_nodes, num_edges) == "node":
                features[feat_name] = _feature_info_from_tensor(value)
        return features

    def get_edge_features(self) -> dict[str, FeatureInfo]:
        """エッジ特徴量辞書を返す。

        Returns:
            dict[str, FeatureInfo]: エッジ特徴量辞書。
        """
        num_nodes = self._resolve_num_nodes().inferred
        num_edges = _get_pyg_edge_count(getattr(self._graph, "edge_index", None))
        features: dict[str, FeatureInfo] = {}
        for feat_name, value in _iter_store_items(self._graph):
            if _classify_pyg_homo_feature(feat_name, value, num_nodes, num_edges) == "edge":
                features[feat_name] = _feature_info_from_tensor(value)
        return features

    def get_degree_stats(self) -> dict[str, NodeDegreeStats]:
        """次数統計辞書を返す。

        Returns:
            dict[str, NodeDegreeStats]: homogeneous 用次数統計辞書。
        """
        num_nodes = self._resolve_num_nodes().inferred
        edge_index = getattr(self._graph, "edge_index", None)
        if num_nodes is None:
            return {}
        return {"_N": _compute_pyg_degree_stats(edge_index, num_nodes)}

    def get_warnings(self) -> list[str]:
        """表示向け warning 一覧を返す。

        Returns:
            list[str]: warning 一覧。
        """
        resolution = self._resolve_num_nodes()
        if resolution.explicit is not None:
            return []
        if resolution.inferred is not None and resolution.source is not None:
            return [
                "num_nodes is not explicitly defined; "
                f"inferred node count {resolution.inferred} from `{resolution.source}` "
                "for feature and degree analysis."
            ]
        return [
            "num_nodes is not explicitly defined; feature and degree analysis may be incomplete."
        ]


def _infer_pyg_hetero_store_num_nodes(store: object) -> NodeCountResolution:
    """PyG heterogeneous ノード store の node count を解決する。

    Args:
        store: ノード store。

    Returns:
        NodeCountResolution: node count の解決結果。
    """
    explicit = _get_explicit_num_nodes(store)
    if explicit is not None:
        return NodeCountResolution(explicit=explicit, inferred=explicit, source="num_nodes")

    x = getattr(store, "x", None)
    if _is_torch_tensor(x) and x.dim() > 0:
        return NodeCountResolution(explicit=None, inferred=int(x.size(0)), source="x")

    for name, value in _iter_store_items(store):
        if not _is_torch_tensor(value):
            continue
        if value.dim() == 0 or name in {"adj_t", "ptr"}:
            continue
        return NodeCountResolution(explicit=None, inferred=int(value.size(0)), source=name)

    return NodeCountResolution(explicit=None, inferred=None, source=None)


class PygHeteroDataInfoAdapter(GraphInfoBackendAdapter):
    """PyG heterogeneous グラフ向けの info アダプタ。"""

    @property
    def backend_name(self) -> str:
        """backend 名を返す。

        Returns:
            str: `"pyg"`。
        """
        return "pyg"

    @property
    def graph_type(self) -> str:
        """グラフ種別を返す。

        Returns:
            str: `"heterogeneous"`。
        """
        return "heterogeneous"

    def _resolve_num_nodes(self) -> dict[str, NodeCountResolution]:
        """ノードタイプごとの node count 解決結果を返す。

        Returns:
            dict[str, NodeCountResolution]: node type ごとの解決結果。
        """
        graph = self._graph
        resolutions = {
            ntype: _infer_pyg_hetero_store_num_nodes(graph[ntype]) for ntype in graph.node_types
        }

        inferred_from_edges: dict[str, int] = {}
        for src_type, rel_type, dst_type in graph.edge_types:
            edge_index = graph[(src_type, rel_type, dst_type)].edge_index
            if not _is_torch_tensor(edge_index) or edge_index.numel() == 0:
                continue
            inferred_from_edges[src_type] = max(
                inferred_from_edges.get(src_type, 0),
                int(edge_index[0].max().item()) + 1,
            )
            inferred_from_edges[dst_type] = max(
                inferred_from_edges.get(dst_type, 0),
                int(edge_index[1].max().item()) + 1,
            )

        for ntype, resolution in resolutions.items():
            if resolution.inferred is None and ntype in inferred_from_edges:
                resolutions[ntype] = NodeCountResolution(
                    explicit=None,
                    inferred=inferred_from_edges[ntype],
                    source="edge_index",
                )

        return resolutions

    def get_num_nodes(self) -> dict[str, int | None]:
        """ノード数辞書を返す。

        Returns:
            dict[str, int]: ノードタイプごとのノード数。
        """
        resolutions = self._resolve_num_nodes()
        return {ntype: resolution.explicit for ntype, resolution in resolutions.items()}

    def get_num_edges(self) -> dict[str, int]:
        """エッジ数辞書を返す。

        Returns:
            dict[str, int]: エッジタイプごとのエッジ数。
        """
        graph = self._graph
        return {
            f"{src_type}->{dst_type}": _get_pyg_edge_count(graph[(src_type, rel_type, dst_type)].edge_index)
            for src_type, rel_type, dst_type in graph.edge_types
        }

    def get_node_features(self) -> dict[str, FeatureInfo]:
        """ノード特徴量辞書を返す。

        Returns:
            dict[str, FeatureInfo]: ノード特徴量辞書。
        """
        graph = self._graph
        resolutions = self._resolve_num_nodes()
        features: dict[str, FeatureInfo] = {}
        for ntype in graph.node_types:
            store = graph[ntype]
            node_count = resolutions[ntype].inferred
            for feat_name, value in _iter_store_items(store):
                if _classify_pyg_hetero_feature(feat_name, value, node_count):
                    features[f"{ntype}.{feat_name}"] = _feature_info_from_tensor(value)
        return features

    def get_edge_features(self) -> dict[str, FeatureInfo]:
        """エッジ特徴量辞書を返す。

        Returns:
            dict[str, FeatureInfo]: エッジ特徴量辞書。
        """
        graph = self._graph
        features: dict[str, FeatureInfo] = {}
        for src_type, rel_type, dst_type in graph.edge_types:
            store = graph[(src_type, rel_type, dst_type)]
            edge_count = _get_pyg_edge_count(store.edge_index)
            for feat_name, value in _iter_store_items(store):
                if _classify_pyg_hetero_feature(feat_name, value, edge_count):
                    features[f"{src_type}->{dst_type}.{feat_name}"] = _feature_info_from_tensor(value)
        return features

    def get_degree_stats(self) -> dict[str, NodeDegreeStats]:
        """次数統計辞書を返す。

        Returns:
            dict[str, NodeDegreeStats]: ノード種別ごとの次数統計。
        """
        import torch

        graph = self._graph
        resolutions = self._resolve_num_nodes()
        inferred_num_nodes = {
            ntype: resolution.inferred for ntype, resolution in resolutions.items()
        }
        in_degree_values = {
            ntype: [0] * count
            for ntype, count in inferred_num_nodes.items()
            if count is not None
        }
        out_degree_values = {
            ntype: [0] * count
            for ntype, count in inferred_num_nodes.items()
            if count is not None
        }

        for src_type, rel_type, dst_type in graph.edge_types:
            store = graph[(src_type, rel_type, dst_type)]
            edge_index = store.edge_index
            if not _is_torch_tensor(edge_index) or edge_index.numel() == 0:
                continue
            if inferred_num_nodes.get(src_type) is None or inferred_num_nodes.get(dst_type) is None:
                continue

            src_counts = _to_int_list(
                torch.bincount(edge_index[0], minlength=inferred_num_nodes[src_type])
            )
            dst_counts = _to_int_list(
                torch.bincount(edge_index[1], minlength=inferred_num_nodes[dst_type])
            )

            out_degree_values[src_type] = [
                current + delta for current, delta in zip(out_degree_values[src_type], src_counts)
            ]
            in_degree_values[dst_type] = [
                current + delta for current, delta in zip(in_degree_values[dst_type], dst_counts)
            ]

        return {
            ntype: NodeDegreeStats(
                in_degree=_build_degree_stats(in_degree_values[ntype]),
                out_degree=_build_degree_stats(out_degree_values[ntype]),
            )
            for ntype in graph.node_types
            if ntype in in_degree_values and ntype in out_degree_values
        }

    def get_warnings(self) -> list[str]:
        """表示向け warning 一覧を返す。

        Returns:
            list[str]: warning 一覧。
        """
        warnings: list[str] = []
        for ntype, resolution in self._resolve_num_nodes().items():
            if resolution.explicit is not None:
                continue
            if resolution.inferred is not None and resolution.source is not None:
                warnings.append(
                    f"{ntype}.num_nodes is not explicitly defined; "
                    f"inferred node count {resolution.inferred} from `{resolution.source}` "
                    "for feature and degree analysis."
                )
            else:
                warnings.append(
                    f"{ntype}.num_nodes is not explicitly defined; "
                    "feature and degree analysis may be incomplete."
                )
        return warnings


def _is_dgl_graph(graph: object) -> bool:
    """DGL グラフらしさを属性ベースで判定する。

    Args:
        graph: 判定対象オブジェクト。

    Returns:
        bool: DGL グラフ互換なら True。
    """
    return all(
        hasattr(graph, attr)
        for attr in ("is_homogeneous", "num_nodes", "num_edges", "in_degrees", "out_degrees")
    )


def _is_pyg_heterodata(graph: object) -> bool:
    """PyG heterogeneous グラフらしさを属性ベースで判定する。

    Args:
        graph: 判定対象オブジェクト。

    Returns:
        bool: HeteroData 互換なら True。
    """
    return all(hasattr(graph, attr) for attr in ("node_types", "edge_types", "__getitem__"))


def _is_pyg_data(graph: object) -> bool:
    """PyG homogeneous グラフらしさを属性ベースで判定する。

    Args:
        graph: 判定対象オブジェクト。

    Returns:
        bool: Data 互換なら True。
    """
    return hasattr(graph, "edge_index") and callable(getattr(graph, "items", None))


def create_graph_info_adapter(graph: object, graphs_count: int = 1) -> GraphInfoBackendAdapter:
    """グラフオブジェクトから適切な adapter を生成する。

    Args:
        graph: backend 固有のグラフオブジェクト。
        graphs_count: ファイル内のグラフ総数。

    Returns:
        GraphInfoBackendAdapter: backend ごとのアダプタ。

    Raises:
        UnsupportedGraphError: サポート対象外のグラフだった場合。
    """
    if _is_dgl_graph(graph):
        return DglInfoAdapter(graph, graphs_count)
    if _is_pyg_heterodata(graph):
        return PygHeteroDataInfoAdapter(graph, graphs_count)
    if _is_pyg_data(graph):
        return PygDataInfoAdapter(graph, graphs_count)
    raise UnsupportedGraphError("unsupported graph object")
