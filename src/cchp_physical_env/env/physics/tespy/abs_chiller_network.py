# Ref: docs/spec/task.md (Task-ID: 010)
# Ref: docs/spec/architecture.md (Pattern: Physics Layer / TESPy)
from __future__ import annotations

from dataclasses import dataclass, field


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass(slots=True)
class AbsChillerDesignPoint:
    q_drive_cap_mw: float = 5.0
    q_cool_cap_mw: float = 4.5
    t_drive_min_k: float = 358.15
    t_drive_ref_k: float = 378.15
    cop_nominal: float = 0.75
    cop_min_fraction: float = 0.50


@dataclass(slots=True)
class AbsChillerResult:
    q_drive_used_mw: float
    q_cool_mw: float
    cop_abs: float
    violation_flags: dict[str, bool] = field(default_factory=dict)


class AbsChillerNetwork:
    """吸收式制冷机：驱动温度与 COP 曲线映射。"""

    def __init__(self, design: AbsChillerDesignPoint) -> None:
        self.design = design

    def estimate_cop(self, t_hot_k: float) -> float:
        if t_hot_k < self.design.t_drive_min_k:
            return 0.0
        if self.design.t_drive_ref_k <= self.design.t_drive_min_k + 1e-9:
            return max(0.0, self.design.cop_nominal)
        scale = (t_hot_k - self.design.t_drive_min_k) / max(
            1e-6, self.design.t_drive_ref_k - self.design.t_drive_min_k
        )
        scale = _clip(scale, 0.0, 1.0)
        cop_floor = max(0.0, self.design.cop_nominal * self.design.cop_min_fraction)
        return cop_floor + (self.design.cop_nominal - cop_floor) * scale

    def solve(self, *, q_drive_request_mw: float, t_hot_k: float) -> AbsChillerResult:
        requested = max(0.0, q_drive_request_mw)
        if requested <= 1e-9:
            return AbsChillerResult(
                q_drive_used_mw=0.0,
                q_cool_mw=0.0,
                cop_abs=max(0.0, self.estimate_cop(t_hot_k=t_hot_k)),
                violation_flags={
                    "abs_drive_temp_low": False,
                    "abs_drive_clipped": False,
                },
            )
        if t_hot_k < self.design.t_drive_min_k:
            return AbsChillerResult(
                q_drive_used_mw=0.0,
                q_cool_mw=0.0,
                cop_abs=0.0,
                violation_flags={
                    "abs_drive_temp_low": requested > 0.0,
                    "abs_drive_clipped": requested > 0.0,
                },
            )

        q_drive_used = _clip(requested, 0.0, self.design.q_drive_cap_mw)
        cop_abs = self.estimate_cop(t_hot_k=t_hot_k)
        q_cool = min(self.design.q_cool_cap_mw, q_drive_used * cop_abs)
        return AbsChillerResult(
            q_drive_used_mw=q_drive_used,
            q_cool_mw=q_cool,
            cop_abs=cop_abs,
            violation_flags={"abs_drive_clipped": requested > q_drive_used + 1e-9},
        )
