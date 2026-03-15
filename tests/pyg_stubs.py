from dataclasses import dataclass, field

import torch


@dataclass
class FakePygData:
    """PyG `Data` 互換の最小テストスタブ。"""

    edge_index: torch.Tensor
    num_nodes: int | None = None
    attributes: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """属性辞書をオブジェクト属性として公開する。"""
        for key, value in self.attributes.items():
            setattr(self, key, value)

    def items(self) -> list[tuple[str, object]]:
        """PyG `Data.items()` 互換の属性列挙を返す。

        Returns:
            list[tuple[str, object]]: 属性名と値のペア。
        """
        pairs: list[tuple[str, object]] = [("edge_index", self.edge_index)]
        pairs.extend((key, value) for key, value in self.attributes.items())
        return pairs


@dataclass
class FakePygNodeStore:
    """PyG `NodeStorage` 互換の最小テストスタブ。"""

    num_nodes: int | None = None
    attributes: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """属性辞書をオブジェクト属性として公開する。"""
        for key, value in self.attributes.items():
            setattr(self, key, value)

    def items(self) -> list[tuple[str, object]]:
        """PyG `NodeStorage.items()` 互換の属性列挙を返す。

        Returns:
            list[tuple[str, object]]: 属性名と値のペア。
        """
        return list(self.attributes.items())


@dataclass
class FakePygEdgeStore:
    """PyG `EdgeStorage` 互換の最小テストスタブ。"""

    edge_index: torch.Tensor
    attributes: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """属性辞書をオブジェクト属性として公開する。"""
        for key, value in self.attributes.items():
            setattr(self, key, value)

    def items(self) -> list[tuple[str, object]]:
        """PyG `EdgeStorage.items()` 互換の属性列挙を返す。

        Returns:
            list[tuple[str, object]]: 属性名と値のペア。
        """
        pairs: list[tuple[str, object]] = [("edge_index", self.edge_index)]
        pairs.extend((key, value) for key, value in self.attributes.items())
        return pairs


@dataclass
class FakePygHeteroData:
    """PyG `HeteroData` 互換の最小テストスタブ。"""

    node_stores: dict[str, FakePygNodeStore]
    edge_stores: dict[tuple[str, str, str], FakePygEdgeStore]

    @property
    def node_types(self) -> list[str]:
        """ノードタイプ一覧を返す。

        Returns:
            list[str]: ノードタイプ一覧。
        """
        return list(self.node_stores.keys())

    @property
    def edge_types(self) -> list[tuple[str, str, str]]:
        """エッジタイプ一覧を返す。

        Returns:
            list[tuple[str, str, str]]: canonical edge type 一覧。
        """
        return list(self.edge_stores.keys())

    def __getitem__(self, key: str | tuple[str, str, str]) -> FakePygNodeStore | FakePygEdgeStore:
        """ノード store またはエッジ store を返す。

        Args:
            key: ノードタイプまたは canonical edge type。

        Returns:
            FakePygNodeStore | FakePygEdgeStore: 対応する store。
        """
        if isinstance(key, tuple):
            return self.edge_stores[key]
        return self.node_stores[key]
