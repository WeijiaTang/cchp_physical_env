# Ref: docs/spec/task.md (Task-ID: 010)
# Ref: docs/spec/architecture.md (Pattern: Physics Layer / Pyomo)
from __future__ import annotations

import pyomo.environ as pyo


def add_gt_ramp_constraint(
    model: pyo.ConcreteModel,
    *,
    p_gt_prev_mw: float,
    gt_ramp_mw_per_step: float,
    enabled: bool,
) -> None:
    if not enabled:
        return

    model.gt_ramp_constraint = pyo.Constraint(
        expr=pyo.inequality(
            p_gt_prev_mw - gt_ramp_mw_per_step,
            model.p_gt_mw,
            p_gt_prev_mw + gt_ramp_mw_per_step,
        )
    )

