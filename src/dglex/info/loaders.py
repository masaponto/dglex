from collections.abc import Sequence
from dataclasses import dataclass

from dglex.info.backends import UnsupportedGraphError, create_graph_info_adapter


class GraphLoadError(ValueError):
    """サポート対象のグラフファイルを読み込めなかったことを示す例外。"""


@dataclass(frozen=True)
class LoadedGraph:
    """ファイルから読み出したグラフオブジェクトと付随情報。

    Attributes:
        graph: backend 固有のグラフオブジェクト。
        graphs_count: ファイル内のグラフ総数。
    """

    graph: object
    graphs_count: int


def _torch_load(path: str) -> object:
    """`torch.load` を CPU 上で実行する。

    Args:
        path: 読み込むファイルパス。

    Returns:
        object: デシリアライズした Python オブジェクト。
    """
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _extract_first_supported_graph(payload: object) -> LoadedGraph:
    """PyG 由来 payload から最初のグラフオブジェクトを抽出する。

    Args:
        payload: `torch.load()` が返した値。

    Returns:
        LoadedGraph: 先頭グラフと総数。

    Raises:
        UnsupportedGraphError: グラフとして解釈できない場合。
    """
    try:
        create_graph_info_adapter(payload)
        return LoadedGraph(graph=payload, graphs_count=1)
    except UnsupportedGraphError:
        pass

    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        graphs = list(payload)
        if not graphs:
            raise UnsupportedGraphError("graph sequence is empty")

        create_graph_info_adapter(graphs[0])
        return LoadedGraph(graph=graphs[0], graphs_count=len(graphs))

    raise UnsupportedGraphError("unsupported serialized graph payload")


def load_graph_from_path(path: str) -> LoadedGraph:
    """ファイルパスからサポート対象のグラフオブジェクトを読み込む。

    Args:
        path: 読み込むファイルパス。

    Returns:
        LoadedGraph: 先頭グラフと総数。

    Raises:
        GraphLoadError: DGL / PyG いずれでも読み込めない場合。
    """
    try:
        import dgl

        graphs, _ = dgl.load_graphs(path)
        if graphs:
            return LoadedGraph(graph=graphs[0], graphs_count=len(graphs))
    except Exception:
        pass

    try:
        payload = _torch_load(path)
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("torch_geometric"):
            raise GraphLoadError("PyG support requires torch-geometric to be installed") from exc
        raise GraphLoadError("file does not contain a supported DGL or PyG graph") from exc
    except Exception as exc:
        raise GraphLoadError("file does not contain a supported DGL or PyG graph") from exc

    try:
        return _extract_first_supported_graph(payload)
    except UnsupportedGraphError as exc:
        raise GraphLoadError("file does not contain a supported DGL or PyG graph") from exc
