# Ref: docs/spec/task.md
# 导入类型注解支持
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping

import numpy as np

# 默认序列观测特征键元组，定义了环境状态观测中包含的特征变量
DEFAULT_SEQUENCE_OBSERVATION_FEATURE_KEYS = (
    # 目标：尽量与 SB3 OBS_KEYS 对齐，避免序列深度策略“看不到关键状态”。
    # 注：旧 checkpoint 会在 metadata 中保存 feature_keys；评估时应以 metadata 为准，
    # 因此修改默认值不会破坏旧模型（见 pipeline/runner.py 的兼容处理）。
    "p_dem_mw",  # 电力需求（兆瓦）
    "qh_dem_mw",  # 热力需求（兆瓦）
    "qc_dem_mw",  # 冷却需求（兆瓦）
    "pv_mw",  # 光伏发电量（兆瓦）
    "wt_mw",  # 风力发电量（兆瓦）
    "price_e",  # 电价
    "price_gas",  # 气价
    "carbon_tax",  # 碳税
    "t_amb_k",  # 环境温度（K）
    "sp_pa",  # 气压（Pa）
    "rh_pct",  # 相对湿度（%）
    "wind_speed",  # 风速
    "wind_direction",  # 风向
    "ghi_wm2",  # 总辐照度
    "dni_wm2",  # 直射辐照度
    "dhi_wm2",  # 散射辐照度
    "soc_bes",  # 储能系统 SOC
    "gt_on",  # GT 是否开启（上一时刻）
    "gt_state",  # GT 状态（0/1/2）
    "e_tes_mwh",  # TES 能量（MWh）
    "t_tes_hot_k",  # TES 热端温度（K）
    "sin_t",  # 日周期 sin
    "cos_t",  # 日周期 cos
    "sin_week",  # 周周期 sin
    "cos_week",  # 周周期 cos
)

# 默认序列动作键元组，定义了控制动作的变量名称
DEFAULT_SEQUENCE_ACTION_KEYS = ("u_gt", "u_bes", "u_boiler", "u_abs", "u_ech", "u_tes")

# 支持的序列适配器类型列表
SUPPORTED_SEQUENCE_ADAPTERS = ("rule", "mlp", "transformer", "mamba")

# 动作边界字典，定义了各控制动作的取值范围限制
_ACTION_BOUNDS = {
    "u_gt": (-1.0, 1.0),      # 燃气轮机控制信号范围
    "u_bes": (-1.0, 1.0),     # 储能系统控制信号范围
    "u_boiler": (0.0, 1.0),   # 锅炉控制信号范围
    "u_abs": (0.0, 1.0),      # 吸收式制冷机控制信号范围
}

def build_feature_vector(observation: dict[str, float], feature_keys: Iterable[str]) -> np.ndarray:
    """
    构建特征向量
    
    该函数根据给定的特征键列表从观测字典中提取对应的值，并将其转换为numpy数组格式的特征向量。
    
    参数:
        observation (dict[str, float]): 包含观测数据的字典，键为特征名称，值为浮点数类型的特征值
        feature_keys (Iterable[str]): 特征键的可迭代对象，指定需要提取的特征名称列表
    
    返回:
        np.ndarray: 包含按feature_keys顺序排列的特征值的一维numpy数组，数据类型为float64
    """
    # 根据feature_keys从observation字典中提取对应值并转换为浮点数列表
    values = [float(observation[key]) for key in feature_keys]
    return np.asarray(values, dtype=np.float64)


def build_action_vector(
    action: Mapping[str, float] | None, action_keys: Iterable[str]
) -> np.ndarray:
    action_mapping = action or {}
    values = [float(action_mapping.get(key, 0.0)) for key in action_keys]
    return np.asarray(values, dtype=np.float64)


def normalized_action_vector_to_env_action_dict(
    action_vector: np.ndarray | list[float] | tuple[float, ...],
    *,
    action_keys: tuple[str, ...] = DEFAULT_SEQUENCE_ACTION_KEYS,
) -> dict[str, float]:
    """把 [-1,1] 标准化动作向量映射到环境动作字典。"""

    vector = np.asarray(action_vector, dtype=np.float64).reshape(-1)
    if vector.shape[0] != len(action_keys):
        raise ValueError(
            f"动作向量维度不匹配，期望 {len(action_keys)}，当前 {vector.shape[0]}"
        )
    clipped = np.clip(vector, -1.0, 1.0)
    output: dict[str, float] = {}
    for index, key in enumerate(action_keys):
        value = float(clipped[index])
        if key in {"u_boiler", "u_abs", "u_ech"}:
            output[key] = float((value + 1.0) * 0.5)
        else:
            output[key] = value
    return output


def build_torch_module_predictor(
    *,
    model,
    device: str = "cpu",
    action_keys: tuple[str, ...] = DEFAULT_SEQUENCE_ACTION_KEYS,
) -> Callable[[np.ndarray, dict[str, float]], Mapping[str, float]]:
    """将 torch 模型绑定为 sequence predictor。"""

    try:
        import torch
    except ModuleNotFoundError as error:  # pragma: no cover
        raise ModuleNotFoundError("未检测到 torch，无法构造 predictor。") from error

    target_device = torch.device(device)
    model = model.to(target_device)
    model.eval()

    def predictor(window: np.ndarray, observation: dict[str, float]) -> Mapping[str, float]:
        del observation  # 当前模型仅使用窗口特征，保留参数以对齐协议。
        window_tensor = torch.as_tensor(
            window, dtype=torch.float32, device=target_device
        ).unsqueeze(0)
        with torch.no_grad():
            output = model(window_tensor)
        if isinstance(output, Mapping):
            if "action_mean" in output:
                action_tensor = output["action_mean"]
            else:
                raise KeyError("predictor 输出 dict 但缺少 `action_mean`。")
        else:
            action_tensor = output
        action_vector = (
            action_tensor.squeeze(0).detach().cpu().numpy().astype(np.float64)
        )
        return normalized_action_vector_to_env_action_dict(
            action_vector, action_keys=action_keys
        )

    return predictor


@dataclass(slots=True)
class SequenceWindowBuffer:
    """序列窗口缓冲区类
    
    用于维护一个固定长度的历史状态 - 动作序列窗口，支持滚动更新。
    该类将观测特征和动作特征拼接成统一的特征向量，并维护最近 history_steps 步的完整记录。
    
    Attributes:
        history_steps (int): 历史步数，即序列窗口的长度，必须大于 0。
        feature_keys (tuple[str, ...]): 观测特征的键名元组，用于从观测字典中提取特征。
            默认为 DEFAULT_SEQUENCE_OBSERVATION_FEATURE_KEYS。
        action_feature_keys (tuple[str, ...]): 动作特征的键名元组，用于从动作字典中提取特征。
            默认为 DEFAULT_SEQUENCE_ACTION_KEYS。
        _buffer (np.ndarray | None): 内部缓冲区，存储形状为 (history_steps, n_features) 的 numpy 数组。
    """
    
    history_steps: int
    feature_keys: tuple[str, ...] = DEFAULT_SEQUENCE_OBSERVATION_FEATURE_KEYS
    action_feature_keys: tuple[str, ...] = DEFAULT_SEQUENCE_ACTION_KEYS
    _buffer: np.ndarray | None = None

    def __post_init__(self) -> None:
        """数据类初始化后处理
        
        验证 history_steps 参数的有效性，确保其值大于 0。
        
        Raises:
            ValueError: 当 history_steps 小于或等于 0 时抛出此异常。
        """
        if self.history_steps <= 0:
            raise ValueError("history_steps 必须 > 0。")

    @property
    def n_observation_features(self) -> int:
        """获取观测特征的数量
        
        Returns:
            int: 观测特征键的数量，即 feature_keys 的长度。
        """
        return len(self.feature_keys)

    @property
    def n_action_features(self) -> int:
        """获取动作特征的数量
        
        Returns:
            int: 动作特征键的数量，即 action_feature_keys 的长度。
        """
        return len(self.action_feature_keys)

    @property
    def n_features(self) -> int:
        """获取总特征数量
        
        Returns:
            int: 观测特征数量与动作特征数量之和。
        """
        return self.n_observation_features + self.n_action_features

    def _build_step_vector(
        self,
        observation: dict[str, float],
        previous_action: Mapping[str, float] | None = None,
    ) -> np.ndarray:
        """构建单步特征向量
        
        将观测字典和动作字典转换为数值向量，并将它们拼接成完整的特征向量。
        观测特征在前，动作特征在后。
        
        Args:
            observation (dict[str, float]): 当前时刻的观测字典，键为特征名称，值为浮点数值。
            previous_action (Mapping[str, float] | None, optional): 上一时刻的动作字典。
                如果为 None，则动作部分将使用默认值填充。默认为 None。
        
        Returns:
            np.ndarray: 一维 numpy 数组，包含拼接后的观测特征向量和动作特征向量。
        """
        obs_vector = build_feature_vector(observation, self.feature_keys)
        action_vector = build_action_vector(previous_action, self.action_feature_keys)
        return np.concatenate([obs_vector, action_vector], axis=0)

    def reset(
        self,
        observation: dict[str, float],
        previous_action: Mapping[str, float] | None = None,
    ) -> np.ndarray:
        """重置序列窗口缓冲区
        
        使用当前的观测和动作初始化缓冲区，将所有历史步填充为相同的初始值。
        这通常用于环境重置或序列开始时的初始化。
        
        Args:
            observation (dict[str, float]): 当前时刻的观测字典，键为特征名称，值为浮点数值。
            previous_action (Mapping[str, float] | None, optional): 上一时刻的动作字典。
                如果为 None，则动作部分将使用默认值填充。默认为 None。
        
        Returns:
            np.ndarray: 初始化后的缓冲区副本，形状为 (history_steps, n_features)。
        """
        vector = self._build_step_vector(observation, previous_action=previous_action)
        self._buffer = np.tile(vector.reshape(1, -1), (self.history_steps, 1))
        return self._buffer.copy()

    def push(
        self,
        observation: dict[str, float],
        previous_action: Mapping[str, float] | None = None,
    ) -> np.ndarray:
        """推入新的时间步数据
        
        将新的观测 - 动作对添加到序列窗口中，采用滑动窗口机制：
        - 移除最旧的时间步（第一行）
        - 添加最新的时间步到末尾
        如果缓冲区尚未初始化，则自动调用 reset 进行初始化。
        
        Args:
            observation (dict[str, float]): 当前时刻的观测字典，键为特征名称，值为浮点数值。
            previous_action (Mapping[str, float] | None, optional): 上一时刻的动作字典。
                如果为 None，则动作部分将使用默认值填充。默认为 None。
        
        Returns:
            np.ndarray: 更新后的缓冲区副本，形状为 (history_steps, n_features)。
        """
        vector = self._build_step_vector(
            observation, previous_action=previous_action
        ).reshape(1, -1)
        if self._buffer is None:
            return self.reset(observation, previous_action=previous_action)
        self._buffer = np.vstack([self._buffer[1:], vector])
        return self._buffer.copy()

    def current_window(self) -> np.ndarray:
        """获取当前的序列窗口
        
        返回当前缓冲区的完整副本，包含所有历史步的特征向量。
        
        Returns:
            np.ndarray: 当前缓冲区副本，形状为 (history_steps, n_features)。
        
        Raises:
            RuntimeError: 如果在调用此方法前未调用 reset 进行初始化，则抛出此异常。
        """
        if self._buffer is None:
            raise RuntimeError("序列窗口尚未初始化，请先调用 reset。")
        return self._buffer.copy()


class SequenceAdapter(ABC):
    """序列策略统一协议：输入 K 步观测/动作序列，输出当前动作。"""

    def __init__(
        self,
        *,
        history_steps: int,
        observation_feature_keys: tuple[str, ...] = DEFAULT_SEQUENCE_OBSERVATION_FEATURE_KEYS,
        action_feature_keys: tuple[str, ...] = DEFAULT_SEQUENCE_ACTION_KEYS,
    ) -> None:
        self.history_steps = int(history_steps)
        if self.history_steps <= 0:
            raise ValueError("history_steps 必须 > 0。")

        self.observation_feature_keys = tuple(observation_feature_keys)
        self.action_feature_keys = tuple(action_feature_keys)
        self.window_buffer = SequenceWindowBuffer(
            history_steps=self.history_steps,
            feature_keys=self.observation_feature_keys,
            action_feature_keys=self.action_feature_keys,
        )
        self.previous_action = self._zero_action()

    def _zero_action(self) -> dict[str, float]:
        return {key: 0.0 for key in self.action_feature_keys}

    def _normalize_action(self, action: Mapping[str, float]) -> dict[str, float]:
        normalized: dict[str, float] = {}
        for key in self.action_feature_keys:
            value = float(action.get(key, 0.0))
            if not np.isfinite(value):
                raise ValueError(f"动作 {key} 非有限数值: {value}")
            lower, upper = _ACTION_BOUNDS.get(key, (-1.0, 1.0))
            normalized[key] = float(np.clip(value, lower, upper))
        return normalized

    def reset_episode(self, initial_observation: dict[str, float]) -> None:
        self.previous_action = self._zero_action()
        self.window_buffer.reset(
            initial_observation, previous_action=self.previous_action
        )

    def act(self, observation: dict[str, float]) -> dict[str, float]:
        window = self.window_buffer.push(
            observation, previous_action=self.previous_action
        )
        raw_action = self.predict_action(window=window, observation=observation)
        action = self._normalize_action(raw_action)
        self.previous_action = action
        return action

    @abstractmethod
    def predict_action(
        self, *, window: np.ndarray, observation: dict[str, float]
    ) -> Mapping[str, float]:
        """子类实现：根据序列窗口预测当前动作。"""


class RuleSequenceAdapter(SequenceAdapter):
    """规则型序列适配器：作为 Transformer/Mamba 的无依赖基线。"""

    def __init__(
        self,
        *,
        train_statistics: dict,
        history_steps: int,
        p_gt_cap_mw: float = 12.0,
        q_ech_cap_mw: float = 6.0,
        observation_feature_keys: tuple[str, ...] = DEFAULT_SEQUENCE_OBSERVATION_FEATURE_KEYS,
        action_feature_keys: tuple[str, ...] = DEFAULT_SEQUENCE_ACTION_KEYS,
    ) -> None:
        super().__init__(
            history_steps=history_steps,
            observation_feature_keys=observation_feature_keys,
            action_feature_keys=action_feature_keys,
        )
        self.train_statistics = train_statistics
        self.p_gt_cap_mw = float(p_gt_cap_mw)
        self.q_ech_cap_mw = float(q_ech_cap_mw)

        required_features = {"p_dem_mw", "qh_dem_mw", "qc_dem_mw", "pv_mw", "wt_mw", "price_e"}
        missing = sorted(required_features - set(self.observation_feature_keys))
        if missing:
            raise ValueError(f"规则序列策略缺少观测特征: {missing}")

        self.observation_index = {
            key: idx for idx, key in enumerate(self.observation_feature_keys)
        }
        stats = self.train_statistics.get("stats", {})
        price_stats = stats.get("price_e", {})
        heat_stats = stats.get("qh_dem_mw", {})
        cool_stats = stats.get("qc_dem_mw", {})
        self.price_low = float(price_stats.get("p05", 0.0))
        self.price_high = float(price_stats.get("p95", 1.0))
        self.heat_med = float(heat_stats.get("p50", 0.0))
        self.cool_med = float(cool_stats.get("p50", 0.0))

    def predict_action(
        self, *, window: np.ndarray, observation: dict[str, float]
    ) -> Mapping[str, float]:
        p_dem_idx = self.observation_index["p_dem_mw"]
        p_pv_idx = self.observation_index["pv_mw"]
        p_wt_idx = self.observation_index["wt_mw"]
        price_idx = self.observation_index["price_e"]
        qh_idx = self.observation_index["qh_dem_mw"]
        qc_idx = self.observation_index["qc_dem_mw"]

        p_dem_smooth = float(np.mean(window[:, p_dem_idx]))
        p_re_smooth = float(np.mean(window[:, p_pv_idx] + window[:, p_wt_idx]))
        price_now = float(window[-1, price_idx])
        qh_smooth = float(np.mean(window[:, qh_idx]))
        qc_smooth = float(np.mean(window[:, qc_idx]))

        soc_bes = float(observation["soc_bes"])
        t_hot_k = float(observation["t_tes_hot_k"])
        e_tes_mwh = float(observation["e_tes_mwh"])

        net_load = max(0.0, p_dem_smooth - p_re_smooth)
        gt_ratio = min(1.0, net_load / max(1e-6, self.p_gt_cap_mw))
        u_gt = gt_ratio * 2.0 - 1.0

        if price_now >= self.price_high and soc_bes > 0.25:
            u_bes = 0.8
        elif price_now <= self.price_low and soc_bes < 0.85:
            u_bes = -0.8
        else:
            u_bes = 0.0

        u_boiler = min(
            1.0,
            max(0.0, (qh_smooth - self.heat_med * 0.6) / max(1e-6, self.heat_med)),
        )
        u_abs = 0.9 if (qc_smooth > self.cool_med * 0.5 and t_hot_k >= 358.15) else 0.0
        u_ech = min(1.0, max(0.0, qc_smooth / max(1e-6, self.q_ech_cap_mw)))

        if qh_smooth > self.heat_med and e_tes_mwh > 2.0:
            u_tes = 0.6
        elif qh_smooth < self.heat_med * 0.5 and e_tes_mwh < 16.0:
            u_tes = -0.5
        else:
            u_tes = 0.0

        return {
            "u_gt": float(u_gt),
            "u_bes": float(u_bes),
            "u_boiler": float(u_boiler),
            "u_abs": float(u_abs),
            "u_ech": float(u_ech),
            "u_tes": float(u_tes),
        }


class TransformerSequenceAdapter(SequenceAdapter):
    """Transformer 适配器壳：后续接入真实模型推理。"""

    def __init__(
        self,
        *,
        history_steps: int,
        predictor: Callable[[np.ndarray, dict[str, float]], Mapping[str, float]] | None = None,
        observation_feature_keys: tuple[str, ...] = DEFAULT_SEQUENCE_OBSERVATION_FEATURE_KEYS,
        action_feature_keys: tuple[str, ...] = DEFAULT_SEQUENCE_ACTION_KEYS,
    ) -> None:
        super().__init__(
            history_steps=history_steps,
            observation_feature_keys=observation_feature_keys,
            action_feature_keys=action_feature_keys,
        )
        self.predictor = predictor

    def predict_action(
        self, *, window: np.ndarray, observation: dict[str, float]
    ) -> Mapping[str, float]:
        if self.predictor is None:
            raise RuntimeError(
                "TransformerSequenceAdapter 尚未绑定真实模型推理器；"
                "请先注入 predictor，或使用 --sequence-adapter rule。"
            )
        return self.predictor(window, observation)


class MLPSequenceAdapter(SequenceAdapter):
    """MLP 适配器壳：通过 predictor 推理动作（接口与 Transformer/Mamba 一致）。"""

    def __init__(
        self,
        *,
        history_steps: int,
        predictor: Callable[[np.ndarray, dict[str, float]], Mapping[str, float]] | None = None,
        observation_feature_keys: tuple[str, ...] = DEFAULT_SEQUENCE_OBSERVATION_FEATURE_KEYS,
        action_feature_keys: tuple[str, ...] = DEFAULT_SEQUENCE_ACTION_KEYS,
    ) -> None:
        super().__init__(
            history_steps=history_steps,
            observation_feature_keys=observation_feature_keys,
            action_feature_keys=action_feature_keys,
        )
        self.predictor = predictor

    def predict_action(
        self, *, window: np.ndarray, observation: dict[str, float]
    ) -> Mapping[str, float]:
        if self.predictor is None:
            raise RuntimeError(
                "MLPSequenceAdapter 尚未绑定真实模型推理器；"
                "请先注入 predictor，或使用 --sequence-adapter rule。"
            )
        return self.predictor(window, observation)


class MambaSequenceAdapter(SequenceAdapter):
    """Mamba 适配器壳：后续接入真实模型推理。"""

    def __init__(
        self,
        *,
        history_steps: int,
        predictor: Callable[[np.ndarray, dict[str, float]], Mapping[str, float]] | None = None,
        observation_feature_keys: tuple[str, ...] = DEFAULT_SEQUENCE_OBSERVATION_FEATURE_KEYS,
        action_feature_keys: tuple[str, ...] = DEFAULT_SEQUENCE_ACTION_KEYS,
    ) -> None:
        super().__init__(
            history_steps=history_steps,
            observation_feature_keys=observation_feature_keys,
            action_feature_keys=action_feature_keys,
        )
        self.predictor = predictor

    def predict_action(
        self, *, window: np.ndarray, observation: dict[str, float]
    ) -> Mapping[str, float]:
        if self.predictor is None:
            raise RuntimeError(
                "MambaSequenceAdapter 尚未绑定真实模型推理器；"
                "请先注入 predictor，或使用 --sequence-adapter rule。"
            )
        return self.predictor(window, observation)


def build_sequence_adapter(
    *,
    adapter_name: str,
    train_statistics: dict,
    history_steps: int,
    observation_feature_keys: tuple[str, ...] = DEFAULT_SEQUENCE_OBSERVATION_FEATURE_KEYS,
    action_feature_keys: tuple[str, ...] = DEFAULT_SEQUENCE_ACTION_KEYS,
    p_gt_cap_mw: float = 12.0,
    q_ech_cap_mw: float = 6.0,
    predictor: Callable[[np.ndarray, dict[str, float]], Mapping[str, float]] | None = None,
) -> SequenceAdapter:
    normalized = adapter_name.strip().lower()
    if normalized == "rule":
        return RuleSequenceAdapter(
            train_statistics=train_statistics,
            history_steps=history_steps,
            p_gt_cap_mw=p_gt_cap_mw,
            q_ech_cap_mw=q_ech_cap_mw,
            observation_feature_keys=observation_feature_keys,
            action_feature_keys=action_feature_keys,
        )
    if normalized == "mlp":
        return MLPSequenceAdapter(
            history_steps=history_steps,
            predictor=predictor,
            observation_feature_keys=observation_feature_keys,
            action_feature_keys=action_feature_keys,
        )
    if normalized == "transformer":
        return TransformerSequenceAdapter(
            history_steps=history_steps,
            predictor=predictor,
            observation_feature_keys=observation_feature_keys,
            action_feature_keys=action_feature_keys,
        )
    if normalized == "mamba":
        return MambaSequenceAdapter(
            history_steps=history_steps,
            predictor=predictor,
            observation_feature_keys=observation_feature_keys,
            action_feature_keys=action_feature_keys,
        )
    raise ValueError(
        f"不支持的 sequence adapter: {adapter_name}，"
        f"支持 {SUPPORTED_SEQUENCE_ADAPTERS}"
    )


class SequenceRulePolicy:
    """兼容旧接口：内部委托给 SequenceAdapter。"""

    def __init__(
        self,
        *,
        train_statistics: dict,
        history_steps: int = 16,
        feature_keys: tuple[str, ...] = DEFAULT_SEQUENCE_OBSERVATION_FEATURE_KEYS,
        p_gt_cap_mw: float = 12.0,
        q_ech_cap_mw: float = 6.0,
        sequence_adapter: str = "rule",
        sequence_predictor: Callable[[np.ndarray, dict[str, float]], Mapping[str, float]] | None = None,
    ) -> None:
        self.sequence_adapter_name = sequence_adapter.strip().lower()
        self.adapter = build_sequence_adapter(
            adapter_name=self.sequence_adapter_name,
            train_statistics=train_statistics,
            history_steps=history_steps,
            observation_feature_keys=feature_keys,
            action_feature_keys=DEFAULT_SEQUENCE_ACTION_KEYS,
            p_gt_cap_mw=p_gt_cap_mw,
            q_ech_cap_mw=q_ech_cap_mw,
            predictor=sequence_predictor,
        )

    def reset_episode(self, initial_observation: dict[str, float]) -> None:
        self.adapter.reset_episode(initial_observation)

    def act(self, observation: dict[str, float]) -> dict[str, float]:
        return self.adapter.act(observation)
