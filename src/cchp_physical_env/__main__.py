# Ref: docs/spec/task.md
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from .core.data import (
    EXPECTED_STEPS_PER_YEAR,
    EVAL_YEAR,
    TRAIN_YEAR,
    compute_training_statistics,
    ensure_frozen_schema_consistency,
    load_exogenous_data,
    summarize_exogenous_data,
)
from .core.config_loader import (
    build_env_config_from_overrides,
    build_training_options,
    load_env_overrides,
    load_training_overrides,
)
from .pipeline.calibration import (
    load_calibration_config,
    run_calibration_search,
    validate_calibration_config,
)
from .pipeline.ablation import run_constraint_ablation
from .pipeline.sequence import SUPPORTED_SEQUENCE_ADAPTERS
from .pipeline.runner import evaluate_baseline, train_baseline
from .policy.sb3 import SB3TrainConfig, evaluate_sb3_policy, train_sb3_policy

DEFAULT_TRAIN_PATH = Path("data/processed/cchp_main_15min_2024.csv")
DEFAULT_EVAL_PATH = Path("data/processed/cchp_main_15min_2025.csv")
DEFAULT_ENV_CONFIG_PATH = Path("src/cchp_physical_env/config/config.yaml")
TRAINING_OPTION_KEYS = (
    "seed",
    "policy",
    "sequence_adapter",
    "history_steps",
    "episode_days",
    "episodes",
    "train_steps",
    "batch_size",
    "update_epochs",
    "lr",
    "device",
    "sb3_enabled",
    "sb3_algo",
    "sb3_total_timesteps",
    "sb3_n_envs",
    "sb3_learning_rate",
    "sb3_batch_size",
    "sb3_gamma",
)


def _resolve_env_config_path(args: argparse.Namespace) -> Path:
    path = getattr(args, "env_config", None)
    return DEFAULT_ENV_CONFIG_PATH if path is None else Path(path)


def _resolve_training_options(args: argparse.Namespace) -> dict:
    config_path = _resolve_env_config_path(args)
    training_overrides = load_training_overrides(config_path)
    resolved = build_training_options(training_overrides)
    for key in TRAINING_OPTION_KEYS:
        if hasattr(args, key):
            resolved[key] = getattr(args, key)
    return build_training_options(resolved)


def _print_summary_block(train_path: Path, eval_path: Path) -> None:
    train_df = load_exogenous_data(train_path)
    eval_df = load_exogenous_data(eval_path)
    ensure_frozen_schema_consistency(train_df, eval_df)

    train_summary = summarize_exogenous_data(train_df)
    eval_summary = summarize_exogenous_data(eval_df)
    payload = {"train": train_summary, "eval": eval_summary}

    if train_summary["n_rows"] != EXPECTED_STEPS_PER_YEAR:
        raise RuntimeError(f"训练集行数错误: {train_summary['n_rows']}")
    if eval_summary["n_rows"] != EXPECTED_STEPS_PER_YEAR:
        raise RuntimeError(f"评估集行数错误: {eval_summary['n_rows']}")

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print("schema_consistency: PASS")


def _command_summary(args: argparse.Namespace) -> None:
    env_overrides = load_env_overrides(_resolve_env_config_path(args))
    build_env_config_from_overrides(env_overrides)
    _print_summary_block(train_path=args.train_path, eval_path=args.eval_path)


def _command_train(args: argparse.Namespace) -> None:
    train_df = load_exogenous_data(args.train_path)
    env_overrides = load_env_overrides(_resolve_env_config_path(args))
    env_config = build_env_config_from_overrides(env_overrides)
    training_options = _resolve_training_options(args)
    if bool(training_options.get("sb3_enabled", False)):
        config = SB3TrainConfig(
            algo=training_options["sb3_algo"],
            total_timesteps=training_options["sb3_total_timesteps"],
            episode_days=training_options["episode_days"],
            n_envs=training_options["sb3_n_envs"],
            learning_rate=training_options["sb3_learning_rate"],
            batch_size=training_options["sb3_batch_size"],
            gamma=training_options["sb3_gamma"],
            seed=training_options["seed"],
            device=training_options["device"],
        )
        result = train_sb3_policy(
            train_df=train_df,
            env_config=env_config,
            config=config,
            run_root=args.run_root,
        )
        print(json.dumps({"mode": "train", "train_year": TRAIN_YEAR, "policy": "sb3", **result}, indent=2, ensure_ascii=False))
        return
    if (
        training_options["policy"] == "sequence_rule"
        and training_options["sequence_adapter"] in {"mlp", "transformer", "mamba"}
    ):
        from .policy.trainer import SequenceTrainerConfig, train_sequence_policy

        train_statistics = compute_training_statistics(train_df)
        trainer_config = SequenceTrainerConfig(
            policy_backbone=training_options["sequence_adapter"],
            history_steps=training_options["history_steps"],
            episode_days=training_options["episode_days"],
            train_steps=training_options["train_steps"],
            batch_size=training_options["batch_size"],
            update_epochs=training_options["update_epochs"],
            lr=training_options["lr"],
            seed=training_options["seed"],
            device=training_options["device"],
        )
        result = train_sequence_policy(
            train_df=train_df,
            train_statistics=train_statistics,
            env_config=env_config,
            trainer_config=trainer_config,
            run_root=args.run_root,
        )
        print(
            json.dumps(
                {
                    "mode": "train",
                    "train_year": TRAIN_YEAR,
                    "policy": training_options["policy"],
                    "sequence_adapter": training_options["sequence_adapter"],
                    **result,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    run_dir = train_baseline(
        train_df=train_df,
        episode_days=training_options["episode_days"],
        episodes=training_options["episodes"],
        policy_name=training_options["policy"],
        history_steps=training_options["history_steps"],
        sequence_adapter=training_options["sequence_adapter"],
        seed=training_options["seed"],
        run_root=args.run_root,
        config=env_config,
    )
    print(
        json.dumps(
            {
                "mode": "train",
                "train_year": TRAIN_YEAR,
                "run_dir": str(run_dir),
                "policy": training_options["policy"],
                "history_steps": training_options["history_steps"],
                "sequence_adapter": training_options["sequence_adapter"],
                "episodes": training_options["episodes"],
                "episode_days": training_options["episode_days"],
                "seed": training_options["seed"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def _command_eval(args: argparse.Namespace) -> None:
    eval_df = load_exogenous_data(args.eval_path)
    env_overrides = load_env_overrides(_resolve_env_config_path(args))
    env_config = build_env_config_from_overrides(env_overrides)
    training_options = _resolve_training_options(args)

    if args.checkpoint is not None:
        try:
            checkpoint_payload = json.loads(Path(args.checkpoint).read_text(encoding="utf-8"))
        except Exception:
            checkpoint_payload = {}
        if isinstance(checkpoint_payload, dict) and checkpoint_payload.get("artifact_type") == "sb3_policy":
            if args.run_dir is None:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                run_dir = Path("runs") / f"{stamp}_eval_sb3_auto"
            else:
                run_dir = Path(args.run_dir)
            summary = evaluate_sb3_policy(
                eval_df=eval_df,
                env_config=env_config,
                checkpoint_json=args.checkpoint,
                run_dir=run_dir,
                seed=training_options["seed"],
                deterministic=True,
                device=training_options["device"],
            )
            print(json.dumps({"mode": "eval", "eval_year": EVAL_YEAR, "run_dir": str(run_dir), "summary": summary}, indent=2, ensure_ascii=False))
            return

    if args.run_dir is not None:
        run_dir = Path(args.run_dir)
    elif args.checkpoint is not None:
        run_dir = Path(args.checkpoint).resolve().parent.parent
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path("runs") / f"{stamp}_eval_only"

    summary = evaluate_baseline(
        eval_df=eval_df,
        run_dir=run_dir,
        policy_name=training_options["policy"],
        history_steps=training_options["history_steps"],
        sequence_adapter=training_options["sequence_adapter"],
        seed=training_options["seed"],
        checkpoint_path=args.checkpoint,
        device=training_options["device"],
        config=env_config,
    )
    output = {"mode": "eval", "eval_year": EVAL_YEAR, "run_dir": str(run_dir), "summary": summary}
    print(json.dumps(output, indent=2, ensure_ascii=False))


def _command_sb3_train(args: argparse.Namespace) -> None:
    train_df = load_exogenous_data(args.train_path)
    env_overrides = load_env_overrides(_resolve_env_config_path(args))
    env_config = build_env_config_from_overrides(env_overrides)

    config = SB3TrainConfig(
        algo=args.algo,
        total_timesteps=args.total_timesteps,
        episode_days=args.episode_days,
        n_envs=args.n_envs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gamma=args.gamma,
        seed=args.seed,
        device=args.device,
    )
    result = train_sb3_policy(
        train_df=train_df,
        env_config=env_config,
        config=config,
        run_root=args.run_root,
    )
    print(json.dumps({"mode": "sb3_train", **result}, indent=2, ensure_ascii=False))


def _command_sb3_eval(args: argparse.Namespace) -> None:
    eval_df = load_exogenous_data(args.eval_path)
    env_overrides = load_env_overrides(_resolve_env_config_path(args))
    env_config = build_env_config_from_overrides(env_overrides)
    summary = evaluate_sb3_policy(
        eval_df=eval_df,
        env_config=env_config,
        checkpoint_json=args.checkpoint,
        run_dir=args.run_dir,
        seed=args.seed,
        deterministic=not args.stochastic,
        device=args.device,
    )
    print(json.dumps({"mode": "sb3_eval", "run_dir": str(args.run_dir), "summary": summary}, indent=2, ensure_ascii=False))

def _command_calibrate(args: argparse.Namespace) -> None:
    train_df = load_exogenous_data(args.train_path)
    eval_df = load_exogenous_data(args.eval_path)
    env_overrides = load_env_overrides(_resolve_env_config_path(args))
    training_options = _resolve_training_options(args)
    config = load_calibration_config(args.config)
    search_block = dict(config.get("search", {}))
    search_block["history_steps"] = int(training_options["history_steps"])
    search_block["sequence_adapter"] = str(training_options["sequence_adapter"]).strip().lower()
    config["search"] = search_block
    validate_calibration_config(config)
    result = run_calibration_search(
        train_df=train_df,
        eval_df=eval_df,
        config=config,
        n_samples=args.n_samples,
        seed=training_options["seed"],
        run_root=args.run_root,
        base_env_overrides=env_overrides,
    )
    output = {
        "mode": "calibrate",
        "train_year": TRAIN_YEAR,
        "eval_year": EVAL_YEAR,
        "config": str(args.config),
        **result,
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


def _command_ablation(args: argparse.Namespace) -> None:
    train_df = load_exogenous_data(args.train_path)
    eval_df = load_exogenous_data(args.eval_path)
    env_overrides = load_env_overrides(_resolve_env_config_path(args))
    training_options = _resolve_training_options(args)
    modes = [item.strip() for item in str(args.modes).split(",") if item.strip()]
    result = run_constraint_ablation(
        train_df=train_df,
        eval_df=eval_df,
        modes=modes,
        policy_name=training_options["policy"],
        history_steps=training_options["history_steps"],
        sequence_adapter=training_options["sequence_adapter"],
        seed=training_options["seed"],
        run_root=args.run_root,
        params_path=args.params,
        base_env_overrides=env_overrides,
    )
    output = {
        "mode": "ablation",
        "train_year": TRAIN_YEAR,
        "eval_year": EVAL_YEAR,
        "sequence_adapter": training_options["sequence_adapter"],
        **result,
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m cchp_physical_env",
        description="CCHP Python-only 数据/训练/评估入口",
    )
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--eval-path", type=Path, default=DEFAULT_EVAL_PATH)
    parser.add_argument("--env-config", type=Path, default=DEFAULT_ENV_CONFIG_PATH)

    subparsers = parser.add_subparsers(dest="command")

    summary_parser = subparsers.add_parser("summary", help="打印 2024/2025 摘要并校验冻结 schema。")
    summary_parser.add_argument(
        "--env-config",
        type=Path,
        default=argparse.SUPPRESS,
        help="环境参数配置文件路径（支持放在子命令后）。",
    )

    train_parser = subparsers.add_parser("train", help="运行 baseline 训练骨架（2024）。")
    train_parser.add_argument("--run-root", type=Path, default=Path("runs"))
    train_parser.add_argument("--episode-days", type=int, default=argparse.SUPPRESS)
    train_parser.add_argument("--episodes", type=int, default=argparse.SUPPRESS)
    train_parser.add_argument(
        "--policy", type=str, default=argparse.SUPPRESS, choices=["rule", "random", "sequence_rule"]
    )
    train_parser.add_argument(
        "--env-config",
        type=Path,
        default=argparse.SUPPRESS,
        help="环境参数配置文件路径（支持放在子命令后）。",
    )
    train_parser.add_argument("--history-steps", type=int, default=argparse.SUPPRESS)
    train_parser.add_argument(
        "--sequence-adapter",
        type=str,
        default=argparse.SUPPRESS,
        choices=list(SUPPORTED_SEQUENCE_ADAPTERS),
        help="当 policy=sequence_rule 时选择序列后端。",
    )
    train_parser.add_argument("--train-steps", type=int, default=argparse.SUPPRESS)
    train_parser.add_argument("--batch-size", type=int, default=argparse.SUPPRESS)
    train_parser.add_argument("--lr", type=float, default=argparse.SUPPRESS)
    train_parser.add_argument("--update-epochs", type=int, default=argparse.SUPPRESS)
    train_parser.add_argument("--device", type=str, default=argparse.SUPPRESS)
    train_parser.add_argument("--seed", type=int, default=argparse.SUPPRESS)
    train_parser.add_argument(
        "--sb3-enabled",
        action="store_true",
        default=argparse.SUPPRESS,
        help="启用 SB3 多算法训练（否则走 baseline/sequence trainer）。",
    )
    train_parser.add_argument("--sb3-algo", type=str, default=argparse.SUPPRESS, choices=["ppo", "sac", "td3", "ddpg"])
    train_parser.add_argument("--sb3-total-timesteps", type=int, default=argparse.SUPPRESS)
    train_parser.add_argument("--sb3-n-envs", type=int, default=argparse.SUPPRESS)
    train_parser.add_argument("--sb3-learning-rate", type=float, default=argparse.SUPPRESS)
    train_parser.add_argument("--sb3-batch-size", type=int, default=argparse.SUPPRESS)
    train_parser.add_argument("--sb3-gamma", type=float, default=argparse.SUPPRESS)

    eval_parser = subparsers.add_parser("eval", help="运行 baseline 评估（固定 2025）。")
    eval_parser.add_argument("--run-dir", type=Path, default=None)
    eval_parser.add_argument("--checkpoint", type=Path, default=None)
    eval_parser.add_argument(
        "--policy", type=str, default=argparse.SUPPRESS, choices=["rule", "random", "sequence_rule"]
    )
    eval_parser.add_argument(
        "--env-config",
        type=Path,
        default=argparse.SUPPRESS,
        help="环境参数配置文件路径（支持放在子命令后）。",
    )
    eval_parser.add_argument("--history-steps", type=int, default=argparse.SUPPRESS)
    eval_parser.add_argument(
        "--sequence-adapter",
        type=str,
        default=argparse.SUPPRESS,
        choices=list(SUPPORTED_SEQUENCE_ADAPTERS),
        help="当 policy=sequence_rule 时选择序列后端。",
    )
    eval_parser.add_argument("--device", type=str, default=argparse.SUPPRESS)
    eval_parser.add_argument("--seed", type=int, default=argparse.SUPPRESS)

    sb3_train_parser = subparsers.add_parser("sb3-train", help="用 SB3 训练 PPO/SAC/TD3/DDPG（Task-011，可选依赖）。")
    sb3_train_parser.add_argument("--run-root", type=Path, default=Path("runs"))
    sb3_train_parser.add_argument("--algo", type=str, choices=["ppo", "sac", "td3", "ddpg"], default="ppo")
    sb3_train_parser.add_argument("--total-timesteps", type=int, default=200_000)
    sb3_train_parser.add_argument("--episode-days", type=int, default=14)
    sb3_train_parser.add_argument("--n-envs", type=int, default=1)
    sb3_train_parser.add_argument("--learning-rate", type=float, default=3e-4)
    sb3_train_parser.add_argument("--batch-size", type=int, default=256)
    sb3_train_parser.add_argument("--gamma", type=float, default=0.99)
    sb3_train_parser.add_argument("--device", type=str, default="auto")
    sb3_train_parser.add_argument("--seed", type=int, default=42)
    sb3_train_parser.add_argument(
        "--env-config",
        type=Path,
        default=argparse.SUPPRESS,
        help="环境参数配置文件路径（支持放在子命令后）。",
    )

    sb3_eval_parser = subparsers.add_parser("sb3-eval", help="用 SB3 checkpoint 跑 2025 年评估（Task-011）。")
    sb3_eval_parser.add_argument("--run-dir", type=Path, required=True)
    sb3_eval_parser.add_argument("--checkpoint", type=Path, required=True, help="sb3-train 产出的 baseline_policy.json")
    sb3_eval_parser.add_argument("--device", type=str, default="auto")
    sb3_eval_parser.add_argument("--seed", type=int, default=42)
    sb3_eval_parser.add_argument("--stochastic", action="store_true", help="使用随机动作采样（默认 deterministic）。")
    sb3_eval_parser.add_argument(
        "--env-config",
        type=Path,
        default=argparse.SUPPRESS,
        help="环境参数配置文件路径（支持放在子命令后）。",
    )

    calibrate_parser = subparsers.add_parser("calibrate", help="运行物理参数标定搜索（Task-002）。")
    calibrate_parser.add_argument(
        "--config",
        type=Path,
        default=Path("docs/spec/calibration_config.json"),
        help="标定配置 JSON",
    )
    calibrate_parser.add_argument("--n-samples", type=int, default=6)
    calibrate_parser.add_argument(
        "--env-config",
        type=Path,
        default=argparse.SUPPRESS,
        help="环境参数配置文件路径（支持放在子命令后）。",
    )
    calibrate_parser.add_argument("--history-steps", type=int, default=argparse.SUPPRESS)
    calibrate_parser.add_argument(
        "--sequence-adapter",
        type=str,
        default=argparse.SUPPRESS,
        choices=list(SUPPORTED_SEQUENCE_ADAPTERS),
        help="当 search.policy=sequence_rule 时选择序列后端。",
    )
    calibrate_parser.add_argument("--seed", type=int, default=argparse.SUPPRESS)
    calibrate_parser.add_argument("--run-root", type=Path, default=Path("runs"))

    ablation_parser = subparsers.add_parser("ablation", help="运行约束方式消融（Task-003）。")
    ablation_parser.add_argument(
        "--modes",
        type=str,
        default="physics_in_loop,reward_only",
        help="逗号分隔，例如 physics_in_loop,reward_only",
    )
    ablation_parser.add_argument(
        "--policy", type=str, default=argparse.SUPPRESS, choices=["rule", "random", "sequence_rule"]
    )
    ablation_parser.add_argument(
        "--env-config",
        type=Path,
        default=argparse.SUPPRESS,
        help="环境参数配置文件路径（支持放在子命令后）。",
    )
    ablation_parser.add_argument("--history-steps", type=int, default=argparse.SUPPRESS)
    ablation_parser.add_argument(
        "--sequence-adapter",
        type=str,
        default=argparse.SUPPRESS,
        choices=list(SUPPORTED_SEQUENCE_ADAPTERS),
        help="当 policy=sequence_rule 时选择序列后端。",
    )
    ablation_parser.add_argument("--seed", type=int, default=argparse.SUPPRESS)
    ablation_parser.add_argument("--run-root", type=Path, default=Path("runs"))
    ablation_parser.add_argument("--params", type=Path, default=None, help="可选参数覆盖 JSON")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    command = args.command or "summary"
    if command == "summary":
        _command_summary(args)
        return
    if command == "train":
        _command_train(args)
        return
    if command == "eval":
        _command_eval(args)
        return
    if command == "sb3-train":
        _command_sb3_train(args)
        return
    if command == "sb3-eval":
        _command_sb3_eval(args)
        return
    if command == "calibrate":
        _command_calibrate(args)
        return
    if command == "ablation":
        _command_ablation(args)
        return
    raise ValueError(f"未知命令: {command}")


if __name__ == "__main__":
    main()
