from typing_extensions import Self

from .node import Node


class Edge:
    def __init__(
        self: Self,
        start_node: Node,
        end_node: Node,
        directed: bool,
        weighted: bool,
        weight: int = 1,
    ) -> None:
        self._start_node = start_node
        self._end_node = end_node
        self._directed = directed
        self._weighted = weighted
        self._weight = weight

    @property
    def get_start_node(self: Self) -> Node:
        return self._start_node

    @property
    def get_end_node(self: Self) -> Node:
        return self._end_node

    @property
    def get_weight(self: Self) -> int:
        return self._weight

    def __str__(self: Self) -> str:
        if self._directed:
            if self._weighted:
                return f"({self._start_node.get_index}) -- {self._weight} --> ({self._end_node.get_index})"
            return f"({self._start_node.get_index}) --> ({self._end_node.get_index})"
        else:
            if self._weighted:
                return f"({self._start_node.get_index}) -- {self._weight} -- ({self._end_node.get_index})"
            return f"({self._start_node.get_index}) -- ({self._end_node.get_index})"
