"""PID Lagrangian multipliers: suppresses dual oscillation and accelerates CMDP convergence.

Replaces simple integral-only dual ascent with PID control dynamics.
Default gains restore original behaviour (Ki=0.005, Kp=Kd=0).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class PIDLagrangianConfig:
    kp: float = 0.0
    ki: float = 0.005
    kd: float = 0.0
    anti_windup: float = 10.0


@dataclass(slots=True)
class PIDLagrangianState:
    integral: np.ndarray
    prev_error: np.ndarray

    @classmethod
    def zeros(cls, n_constraints: int = 3) -> PIDLagrangianState:
        return cls(
            integral=np.zeros(n_constraints, dtype=np.float32),
            prev_error=np.zeros(n_constraints, dtype=np.float32),
        )


def pid_lagrangian_update(
    lambdas: np.ndarray,
    violation: np.ndarray,
    targets: np.ndarray,
    config: PIDLagrangianConfig,
    state: PIDLagrangianState,
    dual_scale: float,
) -> tuple[np.ndarray, PIDLagrangianState]:
    """PID dual ascent with anti-windup.

    e_t = C(s,a) - b
    lambda = max(0, Kp*e_t + Ki*sum(e_tau) + Kd*(e_t - e_{t-1}))

    Args:
        lambdas: current multipliers, shape (n_constraints,).
        violation: batch-mean constraint costs, shape (n_constraints,).
        targets: constraint violation targets, shape (n_constraints,).
        config: PID gains and anti-windup clamp.
        state: accumulated integral and previous error.
        dual_scale: linear warmup multiplier in [0, 1].

    Returns:
        (new_lambdas, updated_state)
    """
    _violation = np.asarray(violation, dtype=np.float32).reshape(lambdas.shape)
    _targets = np.asarray(targets, dtype=np.float32).reshape(lambdas.shape)

    error = _violation - _targets
    state.integral = np.clip(
        state.integral.astype(np.float64) + error.astype(np.float64),
        0.0,
        float(config.anti_windup),
    ).astype(np.float32)
    derivative = error - state.prev_error

    raw = (
        float(config.kp) * error
        + float(config.ki) * state.integral
        + float(config.kd) * derivative
    )
    new_lambdas = np.maximum(0.0, raw).astype(np.float32) * float(dual_scale)
    state.prev_error = error.copy()
    return new_lambdas, state
