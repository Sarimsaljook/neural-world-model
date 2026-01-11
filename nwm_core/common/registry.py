from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Generic, Optional, TypeVar

T = TypeVar("T")

@dataclass
class Registry(Generic[T]):
    _items: Dict[str, T]

    def __init__(self) -> None:
        self._items = {}

    def register(self, name: str) -> Callable[[T], T]:
        def _wrap(obj: T) -> T:
            if name in self._items:
                raise KeyError(f"Registry already has key: {name}")
            self._items[name] = obj
            return obj
        return _wrap

    def get(self, name: str) -> T:
        if name not in self._items:
            raise KeyError(f"Unknown registry key: {name}")
        return self._items[name]

    def maybe_get(self, name: str) -> Optional[T]:
        return self._items.get(name)

    def keys(self) -> list[str]:
        return sorted(self._items.keys())
