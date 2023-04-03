from typing import Any

from typing_extensions import Self


class Node:
    def __init__(self: Self, index: int, value: Any) -> None:
        self._index = index
        self._value = value

    @property
    def get_index(self: Self) -> int:
        return self._index

    @property
    def get_value(self: Self) -> Any:
        return self._value

    def __lt__(self: Self, other: Self) -> bool:
        return self._index < other._index

    def __eq__(self: Self, other: Self) -> bool:
        return self._index == other._index

    def __hash__(self: Self) -> int:
        return hash(self._index)
