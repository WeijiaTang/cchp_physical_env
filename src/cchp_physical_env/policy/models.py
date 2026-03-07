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
    from transformers import MambaConfig, MambaModel
except (ModuleNotFoundError, ImportError):  # pragma: no cover
    MambaConfig = None
    MambaModel = None


SUPPORTED_POLICY_BACKBONES = ("mlp", "transformer", "mamba")


def _require_torch() -> None:
    if torch is None:
        raise ModuleNotFoundError(
            "未检测到 torch。请先安装 PyTorch（例如 CUDA 12.4："
            "--extra-index-url https://download.pytorch.org/whl/cu124）。"
        )


def _require_mamba() -> None:
    if MambaConfig is None or MambaModel is None:
        raise ModuleNotFoundError(
            "未检测到 Transformers 中的 Mamba 实现。"
            "请安装/升级 transformers，并确保 torch 可用。"
        )


def _build_mamba_config(*, d_model: int, n_layer: int, d_state: int, d_conv: int, expand: int):
    _require_mamba()
    return MambaConfig(
        vocab_size=1,
        hidden_size=int(d_model),
        state_size=int(d_state),
        num_hidden_layers=int(n_layer),
        conv_kernel=int(d_conv),
        expand=int(expand),
        use_cache=False,
        use_mambapy=False,
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

    class MLPPolicyNet(SequencePolicyNetBase):
        """MLP backbone：输入 (B,K,D) flatten 后输出 (B,A) 动作均值。"""

        def __init__(
            self,
            *,
            history_steps: int,
            n_features: int,
            n_actions: int,
            hidden_sizes: tuple[int, ...] = (256, 256, 256),
            dropout: float = 0.0,
        ) -> None:
            super().__init__(n_features=n_features, n_actions=n_actions)
            if history_steps <= 0:
                raise ValueError("history_steps 必须 > 0。")
            self.history_steps = int(history_steps)
            input_dim = int(self.history_steps * self.n_features)

            layers: list[nn.Module] = []
            prev = input_dim
            for width in hidden_sizes:
                width_int = int(width)
                if width_int <= 0:
                    raise ValueError("hidden_sizes 中每个值必须 > 0。")
                layers.append(nn.Linear(prev, width_int))
                layers.append(nn.GELU())
                if dropout > 0.0:
                    layers.append(nn.Dropout(float(dropout)))
                prev = width_int
            layers.append(nn.Linear(prev, self.n_actions))
            self.net = nn.Sequential(*layers)

        def forward(self, window: Tensor) -> Tensor:
            if window.dim() != 3:
                raise ValueError("MLPPolicyNet 输入必须是 3D 张量 (B,K,D)。")
            if int(window.shape[1]) != self.history_steps:
                raise ValueError(
                    f"history_steps 不匹配：模型={self.history_steps}，输入={int(window.shape[1])}"
                )
            flat = window.reshape(window.shape[0], -1)
            action_mean = torch.tanh(self.net(flat))
            return action_mean


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
            self.input_norm = nn.LayerNorm(self.n_features)
            self.input_proj = nn.Linear(self.n_features, d_model)
            self.mamba = MambaModel(
                _build_mamba_config(
                    d_model=int(d_model),
                    n_layer=int(n_layer),
                    d_state=int(d_state),
                    d_conv=int(d_conv),
                    expand=int(expand),
                )
            )
            self.dropout = nn.Dropout(dropout)
            self.output_norm = nn.LayerNorm(d_model)
            self.action_head = nn.Linear(d_model, self.n_actions)

        def forward(self, window: Tensor) -> Tensor:
            if window.dim() != 3:
                raise ValueError("MambaPolicyNet 输入必须是 3D 张量 (B,K,D)。")
            hidden = self.input_proj(self.input_norm(window))
            outputs = self.mamba(inputs_embeds=hidden, use_cache=False, return_dict=True)
            hidden_last = self.dropout(self.output_norm(outputs.last_hidden_state[:, -1, :]))
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


    class MLPPolicyNet(SequencePolicyNetBase):  # pragma: no cover
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
    if normalized == "mlp":
        if "history_steps" not in kwargs:
            raise ValueError("mlp backbone 需要在 model_kwargs 中提供 history_steps。")
        history_steps = int(kwargs.pop("history_steps"))
        return MLPPolicyNet(
            history_steps=history_steps, n_features=n_features, n_actions=n_actions, **kwargs
        )
    if normalized == "transformer":
        return TransformerPolicyNet(
            n_features=n_features, n_actions=n_actions, **kwargs
        )
    if normalized == "mamba":
        return MambaPolicyNet(n_features=n_features, n_actions=n_actions, **kwargs)
    raise ValueError(
        f"不支持的 policy_backbone: {policy_backbone}，支持 {SUPPORTED_POLICY_BACKBONES}"
    )
