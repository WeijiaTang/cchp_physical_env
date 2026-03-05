from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml


@dataclass
class RunCfg:
    env_config: str = "src/cchp_physical_env/config/config.yaml"
    run_root: str = "runs"
    run_prefix: str = "local_cchp"

    train_path: str = "data/processed/cchp_main_15min_2024.csv"
    eval_path: str = "data/processed/cchp_main_15min_2025.csv"

    resume_enabled: bool = True
    resume_checkpoint: str = ""
    skip_train_when_resuming: bool = True

    env_overrides: dict[str, object] = field(default_factory=dict)
    training_overrides: dict[str, object] = field(default_factory=dict)
    eval_cli_overrides: dict[str, object] = field(default_factory=dict)


def _coerce_scalar(value: str) -> object:
    raw = value.strip()
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        if raw.startswith("0") and len(raw) > 1 and raw[1].isdigit():
            raise ValueError
        return int(raw)
    except Exception:
        pass
    try:
        return float(raw)
    except Exception:
        return raw


def _parse_kv_list(items: list[str]) -> dict[str, object]:
    out: dict[str, object] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"无效参数，必须是 key=value: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"无效参数 key: {item}")
        out[key] = _coerce_scalar(value)
    return out


def _load_yaml_mapping(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError("配置文件顶层必须是 mapping")
    return payload


def _write_runtime_config(*, base_config_path: Path, cfg: RunCfg, output_dir: Path) -> Path:
    base_cfg = _load_yaml_mapping(base_config_path)

    env_block = base_cfg.get("env")
    if env_block is None or not isinstance(env_block, dict):
        raise ValueError("config.yaml 缺少 env block 或 env 不是 mapping")
    training_block = base_cfg.get("training")
    if training_block is None or not isinstance(training_block, dict):
        raise ValueError("config.yaml 缺少 training block 或 training 不是 mapping")

    env_block.update(dict(cfg.env_overrides))
    training_block.update(dict(cfg.training_overrides))
    base_cfg["env"] = env_block
    base_cfg["training"] = training_block

    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runtime_path = output_dir / f"{cfg.run_prefix}_runtime_{stamp}.yaml"
    runtime_path.write_text(
        yaml.safe_dump(base_cfg, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return runtime_path


def _run_cli(args: list[str]) -> None:
    env = os.environ.copy()
    py_src = str((Path.cwd() / "src").resolve())
    env["PYTHONPATH"] = py_src + (os.pathsep + env["PYTHONPATH"] if "PYTHONPATH" in env else "")
    cmd = [sys.executable, "-m", "cchp_physical_env", *args]
    subprocess.run(cmd, check=True, env=env)


def _latest_checkpoint_json(run_root: Path) -> Path | None:
    if not run_root.exists():
        return None
    candidates = sorted(run_root.rglob("baseline_policy.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _resolve_checkpoint_from_args(*, cfg: RunCfg, run_root: Path) -> Path | None:
    if cfg.resume_checkpoint:
        p = Path(cfg.resume_checkpoint)
        return p if p.exists() else None
    return _latest_checkpoint_json(run_root)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python main.py")

    p.add_argument("--env-config", default=RunCfg.env_config)
    p.add_argument("--run-root", default=RunCfg.run_root)
    p.add_argument("--run-prefix", default=RunCfg.run_prefix)
    p.add_argument("--train-path", default=RunCfg.train_path)
    p.add_argument("--eval-path", default=RunCfg.eval_path)

    p.add_argument("--resume", dest="resume_enabled", action="store_true", default=RunCfg.resume_enabled)
    p.add_argument("--no-resume", dest="resume_enabled", action="store_false")
    p.add_argument("--resume-checkpoint", default=RunCfg.resume_checkpoint)
    p.add_argument(
        "--skip-train-when-resuming",
        dest="skip_train_when_resuming",
        action="store_true",
        default=RunCfg.skip_train_when_resuming,
    )
    p.add_argument("--no-skip-train-when-resuming", dest="skip_train_when_resuming", action="store_false")

    p.add_argument("--env-override", action="append", default=[])
    p.add_argument("--training-override", action="append", default=[])
    p.add_argument("--eval-override", action="append", default=[])

    p.add_argument("--only", choices=["summary", "train", "eval"], default="")

    return p


def main() -> None:
    args = _build_parser().parse_args()

    cfg = RunCfg(
        env_config=str(args.env_config),
        run_root=str(args.run_root),
        run_prefix=str(args.run_prefix),
        train_path=str(args.train_path),
        eval_path=str(args.eval_path),
        resume_enabled=bool(args.resume_enabled),
        resume_checkpoint=str(args.resume_checkpoint),
        skip_train_when_resuming=bool(args.skip_train_when_resuming),
        env_overrides=_parse_kv_list(list(args.env_override)),
        training_overrides=_parse_kv_list(list(args.training_override)),
        eval_cli_overrides=_parse_kv_list(list(args.eval_override)),
    )

    run_root = Path(cfg.run_root)
    runtime_cfg_path = _write_runtime_config(
        base_config_path=Path(cfg.env_config),
        cfg=cfg,
        output_dir=run_root,
    )

    if not args.only or args.only == "summary":
        _run_cli(
            [
                "--train-path",
                str(Path(cfg.train_path).as_posix()),
                "--eval-path",
                str(Path(cfg.eval_path).as_posix()),
                "--env-config",
                runtime_cfg_path.as_posix(),
                "summary",
            ]
        )
        if args.only == "summary":
            return

    checkpoint_json: Path | None = None
    if cfg.resume_enabled and cfg.skip_train_when_resuming:
        checkpoint_json = _resolve_checkpoint_from_args(cfg=cfg, run_root=run_root)

    if checkpoint_json is None:
        if not args.only or args.only == "train":
            _run_cli(
                [
                    "--train-path",
                    str(Path(cfg.train_path).as_posix()),
                    "--env-config",
                    runtime_cfg_path.as_posix(),
                    "train",
                    "--run-root",
                    run_root.as_posix(),
                ]
            )
            if args.only == "train":
                return
        checkpoint_json = _latest_checkpoint_json(run_root)

    if checkpoint_json is None or not checkpoint_json.exists():
        raise FileNotFoundError("未找到 baseline_policy.json，无法执行 eval")

    if not args.only or args.only == "eval":
        eval_args: list[str] = [
            "--eval-path",
            str(Path(cfg.eval_path).as_posix()),
            "--env-config",
            runtime_cfg_path.as_posix(),
            "eval",
            "--checkpoint",
            checkpoint_json.as_posix(),
        ]
        for key, value in cfg.eval_cli_overrides.items():
            eval_args.extend([f"--{key.replace('_', '-')}", str(value)])
        _run_cli(eval_args)

        payload = {
            "runtime_config": runtime_cfg_path.as_posix(),
            "checkpoint_json": checkpoint_json.as_posix(),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()