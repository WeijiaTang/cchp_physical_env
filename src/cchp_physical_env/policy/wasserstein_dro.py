"""Wasserstein Distributionally Robust critic regularisation.

Adds gradient-penalty on the (observation, action) input to the Q-function,
equivalent to a worst-case Wasserstein-ball DRO dual reformulation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn


def wasserstein_gradient_penalty(
    critic: nn.Module,
    obs: Tensor,
    action: Tensor,
    obs_dim: int,
) -> Tensor:
    """||grad_{(obs,action)} Q(obs,action)||^2 — the Wasserstein DRO penalty."""
    x = torch.cat([obs, action], dim=-1).requires_grad_(True)
    q = critic(x[:, :obs_dim], x[:, obs_dim:])
    grads = torch.autograd.grad(
        q.sum(), x, create_graph=True, retain_graph=True,
    )[0]
    return grads.pow(2).sum(dim=-1).mean()


def wasserstein_dual_critic_loss(
    critic: nn.Module,
    obs: Tensor,
    action: Tensor,
    q_target: Tensor,
    penalty_coef: float,
    obs_dim: int,
) -> tuple[Tensor, Tensor]:
    """MSE critic loss regularised with Wasserstein gradient penalty.

    Returns:
        (total_loss, gradient_penalty_value_detached)
    """
    x = torch.cat([obs, action], dim=-1)
    q_pred = critic(x[:, :obs_dim], x[:, obs_dim:])
    mse = F.mse_loss(q_pred, q_target)
    if penalty_coef <= 0.0:
        return mse, torch.tensor(0.0, device=mse.device)
    grads = torch.autograd.grad(
        q_pred.sum(), x, create_graph=True, retain_graph=True,
    )[0]
    gp = grads.pow(2).sum(dim=-1).mean()
    return mse + float(penalty_coef) * gp, gp.detach()
