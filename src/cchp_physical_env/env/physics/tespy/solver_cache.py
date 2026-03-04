# Ref: docs/spec/task.md (Task-ID: 010)
# Ref: docs/spec/architecture.md (Pattern: Physics Layer / TESPy)
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass(slots=True)
class SolverCache:
    """设计点与离设计点的轻量缓存，减少重复求解。"""

    design_cache: dict[str, dict[str, float]] = field(default_factory=dict)
    offdesign_cache: dict[str, dict[str, float]] = field(default_factory=dict)

    def get_design(self, key: str) -> dict[str, float] | None:
        value = self.design_cache.get(key)
        return dict(value) if value is not None else None

    def set_design(self, key: str, value: dict[str, float]) -> None:
        self.design_cache[key] = dict(value)

    def get_or_compute_design(
        self,
        key: str,
        solver: Callable[[], dict[str, float]],
    ) -> dict[str, float]:
        cached = self.get_design(key)
        if cached is not None:
            return cached
        value = solver()
        self.set_design(key, value)
        return dict(value)

    def get_offdesign(self, key: str) -> dict[str, float] | None:
        value = self.offdesign_cache.get(key)
        return dict(value) if value is not None else None

    def set_offdesign(self, key: str, value: dict[str, float]) -> None:
        self.offdesign_cache[key] = dict(value)

