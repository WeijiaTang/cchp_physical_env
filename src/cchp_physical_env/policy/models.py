# Ref: docs/spec/task.md
from __future__ import annotations

from typing import Any

try:
    import torch
    from torch import Tensor, nn
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    Tensor = Any
    nn = Any

try:
    from mamba_ssm import Mamba as MambaBlock
except ModuleNotFoundError:  # pragma: no cover
    MambaBlock = None


SUPPORTED_POLICY_BACKBONES = ("transformer", "mamba")


def _require_torch() -> None:
    if torch is None:
        raise ModuleNotFoundError(
            "未检测到 torch。请先安装 PyTorch（例如 CUDA 12.4："
            "--extra-index-url https://download.pytorch.org/whl/cu124）。"
        )


def _require_mamba() -> None:
    if MambaBlock is None:
        raise ModuleNotFoundError(
            "未检测到 mamba-ssm。请安装 mamba-ssm>=1.2（通常需 CUDA 环境）。"
        )


if torch is not None:

    class SequencePolicyNetBase(nn.Module):
        """序列策略网络基类：输出标准化动作均值，并提供高斯分布接口。"""

        def __init__(self, n_features: int, n_actions: int) -> None:
            super().__init__()
            if n_features <= 0:
                raise ValueError("n_features 必须 > 0。")
            if n_actions <= 0:
                raise ValueError("n_actions 必须 > 0。")
            self.n_features = int(n_features)
            self.n_actions = int(n_actions)
            self.log_std = nn.Parameter(torch.zeros(self.n_actions, dtype=torch.float32))

        def action_distribution(self, window: Tensor) -> torch.distributions.Normal:
            mean = self.forward(window)
            std = torch.exp(self.log_std).clamp(min=1e-3, max=2.0).expand_as(mean)
            return torch.distributions.Normal(mean, std)


    class TransformerPolicyNet(SequencePolicyNetBase):
        """Transformer backbone：输入 (B,K,D)，输出 (B,A) 动作均值。"""

        def __init__(
            self,
            n_features: int,
            n_actions: int,
            d_model: int = 128,
            n_head: int = 4,
            n_layer: int = 3,
            dropout: float = 0.1,
        ) -> None:
            super().__init__(n_features=n_features, n_actions=n_actions)
            self.input_proj = nn.Linear(self.n_features, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_head,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer=encoder_layer, num_layers=n_layer
            )
            self.output_norm = nn.LayerNorm(d_model)
            self.action_head = nn.Linear(d_model, self.n_actions)

        def forward(self, window: Tensor) -> Tensor:
            if window.dim() != 3:
                raise ValueError("TransformerPolicyNet 输入必须是 3D 张量 (B,K,D)。")
            hidden = self.input_proj(window)
            hidden = self.transformer(hidden)
            hidden_last = self.output_norm(hidden[:, -1, :])
            action_mean = torch.tanh(self.action_head(hidden_last))
            return action_mean


    class MambaPolicyNet(SequencePolicyNetBase):
        """Mamba backbone：输入 (B,K,D)，输出 (B,A) 动作均值。"""

        def __init__(
            self,
            n_features: int,
            n_actions: int,
            d_model: int = 128,
            n_layer: int = 4,
            d_state: int = 16,
            d_conv: int = 4,
            expand: int = 2,
            dropout: float = 0.1,
        ) -> None:
            _require_mamba()
            super().__init__(n_features=n_features, n_actions=n_actions)
            self.input_proj = nn.Linear(self.n_features, d_model)
            self.blocks = nn.ModuleList(
                [
                    MambaBlock(
                        d_model=d_model,
                        d_state=d_state,
                        d_conv=d_conv,
                        expand=expand,
                    )
                    for _ in range(n_layer)
                ]
            )
            self.dropout = nn.Dropout(dropout)
            self.output_norm = nn.LayerNorm(d_model)
            self.action_head = nn.Linear(d_model, self.n_actions)

        def forward(self, window: Tensor) -> Tensor:
            if window.dim() != 3:
                raise ValueError("MambaPolicyNet 输入必须是 3D 张量 (B,K,D)。")
            hidden = self.input_proj(window)
            for block in self.blocks:
                hidden = hidden + self.dropout(block(hidden))
            hidden_last = self.output_norm(hidden[:, -1, :])
            action_mean = torch.tanh(self.action_head(hidden_last))
            return action_mean


else:

    class SequencePolicyNetBase:  # pragma: no cover
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _require_torch()


    class TransformerPolicyNet(SequencePolicyNetBase):  # pragma: no cover
        pass


    class MambaPolicyNet(SequencePolicyNetBase):  # pragma: no cover
        pass


def build_policy_network(
    *,
    policy_backbone: str,
    n_features: int,
    n_actions: int,
    model_kwargs: dict[str, Any] | None = None,
) -> SequencePolicyNetBase:
    """按名称构造策略网络。"""

    _require_torch()
    kwargs = dict(model_kwargs or {})
    normalized = policy_backbone.strip().lower()
    if normalized == "transformer":
        return TransformerPolicyNet(
            n_features=n_features, n_actions=n_actions, **kwargs
        )
    if normalized == "mamba":
        return MambaPolicyNet(n_features=n_features, n_actions=n_actions, **kwargs)
    raise ValueError(
        f"不支持的 policy_backbone: {policy_backbone}，支持 {SUPPORTED_POLICY_BACKBONES}"
    )
