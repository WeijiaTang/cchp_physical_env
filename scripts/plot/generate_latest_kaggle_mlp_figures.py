from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[2]
PLOT_ROOT = ROOT / "scripts" / "plot"
OUTPUT_DIRS = {
    "png": PLOT_ROOT / "png",
    "pdf": PLOT_ROOT / "pdf",
    "eps": PLOT_ROOT / "eps",
    "tiff": PLOT_ROOT / "tiff",
}


@dataclass(frozen=True)
class RunSpec:
    algo: str
    label: str
    color: str
    run_dir: Path


RUN_SPECS: tuple[RunSpec, ...] = (
    RunSpec(
        algo="dqn",
        label="rbDQN",
        color="#1B5E7A",
        run_dir=ROOT
        / "kaggle/CCHP-SB3-dqn+mlp03272030/runs/20260327_032502_853296_train_sb3_dqn_mlp_k32",
    ),
    RunSpec(
        algo="ddpg",
        label="DDPG+rule_residual",
        color="#2D6A4F",
        run_dir=ROOT
        / "kaggle/CCHP-SB3-ddpg+mlp03272030/runs/20260327_032418_930697_train_sb3_ddpg_mlp_k32",
    ),
    RunSpec(
        algo="td3",
        label="TD3+rule_residual",
        color="#B07D3C",
        run_dir=ROOT
        / "kaggle/CCHP-SB3-td3+mlp03272030/runs/20260327_032436_184749_train_sb3_td3_mlp_k32",
    ),
    RunSpec(
        algo="ppo",
        label="PPO+rule_residual",
        color="#B45E5E",
        run_dir=ROOT
        / "kaggle/CCHP-SB3-ppo+mlp03272030/runs/20260327_032544_641642_train_sb3_ppo_mlp_k32",
    ),
    RunSpec(
        algo="sac",
        label="SAC+rule_residual",
        color="#7C6F9B",
        run_dir=ROOT
        / "kaggle/CCHP-SB3-sac+mlp03272030/runs/20260327_032508_703855_train_sb3_sac_mlp_k32",
    ),
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _hex_to_rgb(color: str) -> tuple[float, float, float]:
    value = color.lstrip("#")
    return tuple(int(value[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def lighten_color(color: str, amount: float = 0.35) -> tuple[float, float, float]:
    r, g, b = _hex_to_rgb(color)
    return tuple(1.0 - (1.0 - channel) * (1.0 - amount) for channel in (r, g, b))


def _set_theme() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "axes.edgecolor": "#AAB2BF",
            "axes.labelcolor": "#1F2430",
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 10.5,
            "xtick.color": "#1F2430",
            "ytick.color": "#1F2430",
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "grid.color": "#E7EBF0",
            "grid.linewidth": 0.8,
            "grid.alpha": 1.0,
            "legend.frameon": False,
            "legend.fontsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _style_axes(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    ax.grid(True, axis=grid_axis, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.spines["left"].set_color("#AAB2BF")
    ax.spines["bottom"].set_color("#AAB2BF")


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.07,
        label,
        transform=ax.transAxes,
        fontsize=12.5,
        fontweight="bold",
        color="#1F2430",
        va="top",
    )


def _format_million_steps(value: float, _: float) -> str:
    return f"{value / 1_000_000.0:.1f}M"


def _save_figure(fig: plt.Figure, stem: str) -> None:
    for fmt, out_dir in OUTPUT_DIRS.items():
        out_dir.mkdir(parents=True, exist_ok=True)
        dpi = 600 if fmt in {"png", "tiff"} else 300
        fig.savefig(out_dir / f"{stem}.{fmt}", format=fmt, dpi=dpi)


def _find_low_lr_event(curve: pd.DataFrame) -> float | None:
    if "plateau__event__action" not in curve.columns:
        return None
    hit = curve[curve["plateau__event__action"] == "low_lr_fine_tune"]
    if hit.empty:
        return None
    return float(hit.iloc[0]["timesteps"])


def _load_dataset() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    curves: dict[str, pd.DataFrame] = {}
    meta: dict[str, dict[str, Any]] = {}
    for spec in RUN_SPECS:
        train = _load_json(spec.run_dir / "train" / "summary.json")
        eval_summary = _load_json(spec.run_dir / "eval" / "summary.json")
        behavior = _load_json(spec.run_dir / "eval" / "behavior_metrics.json")
        curve = pd.read_csv(spec.run_dir / "train" / "learning_curve_eval.csv")
        step_log = pd.read_csv(spec.run_dir / "eval" / "step_log_light.csv")

        phys_rate = float((step_log["u_abs_phys"] > 1e-6).mean())
        q_abs_mean = float(step_log["q_abs_cool_mw"].mean())

        rows.append(
            {
                "algo": spec.algo,
                "label": spec.label,
                "color": spec.color,
                "run_dir": str(spec.run_dir),
                "total_cost_m": float(eval_summary["total_cost"]) / 1_000_000.0,
                "rel_heat": float(eval_summary["reliability"]["heat"]),
                "rel_cool": float(eval_summary["reliability"]["cooling"]),
                "unmet_h_mwh": float(eval_summary["unmet_energy_mwh"]["heat"]),
                "unmet_c_mwh": float(eval_summary["unmet_energy_mwh"]["cooling"]),
                "violation_rate": float(eval_summary["violation_rate"]),
                "export_penalty_m": float(eval_summary["cost_breakdown"]["grid_export_penalty"]) / 1_000_000.0,
                "starts_gt": int(eval_summary["starts"]["gt"]),
                "starts_ech": int(eval_summary["starts"]["ech"]),
                "gt_toggle_steps": int(behavior["gt_toggle_steps"]),
                "cool_unmet_steps": int(behavior["cool_unmet_steps"]),
                "export_over_soft_cap_steps": int(behavior["export_over_soft_cap_steps"]),
                "abs_blocked_rate": float(behavior["abs_blocked_rate"]),
                "u_abs_phys_rate": phys_rate,
                "q_abs_mean_mw": q_abs_mean,
                "best_gate_passed": bool(train["best_model_selection"]["gate_passed"]),
                "selected_best_t": float(train["convergence_summary"]["selected_best"]["timesteps"]),
                "reward_best_t": float(train["convergence_summary"]["reward_best"]["timesteps"]),
                "low_lr_event_t": _find_low_lr_event(curve),
                "final_lr": float(train["final_learning_rate"]),
            }
        )
        curves[spec.algo] = curve.copy()
        meta[spec.algo] = {
            "label": spec.label,
            "color": spec.color,
        }

    metrics = pd.DataFrame(rows).sort_values("total_cost_m", ascending=True).reset_index(drop=True)
    order = metrics["algo"].tolist()
    return {"metrics": metrics, "curves": curves, "meta": meta, "order": order}


def plot_main_results(bundle: dict[str, Any]) -> plt.Figure:
    df = bundle["metrics"].copy()
    labels = df["label"].tolist()
    colors = df["color"].tolist()
    x = np.arange(len(df))

    fig, axes = plt.subplots(2, 2, figsize=(14.6, 9.2))
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.12, top=0.86, wspace=0.17, hspace=0.28)
    ax_cost, ax_rel, ax_unmet, ax_burden = axes.flat

    bars = ax_cost.bar(
        x,
        df["total_cost_m"],
        color=[lighten_color(color, 0.10) for color in colors],
        edgecolor=colors,
        linewidth=1.1,
        width=0.68,
        zorder=3,
    )
    for bar, value in zip(bars, df["total_cost_m"]):
        ax_cost.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.08,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#1F2430",
        )
    ax_cost.set_title("Annual total cost")
    ax_cost.set_ylabel("Million CNY-equivalent")
    ax_cost.set_xticks(x, labels, rotation=16, ha="right")
    _style_axes(ax_cost, grid_axis="y")
    _panel_label(ax_cost, "(a)")

    for idx, row in df.iterrows():
        ax_rel.plot(
            [idx, idx],
            [row["rel_cool"], row["rel_heat"]],
            color=row["color"],
            linewidth=2.0,
            zorder=2,
        )
        ax_rel.scatter(
            idx,
            row["rel_heat"],
            s=78,
            facecolor="white",
            edgecolor=row["color"],
            linewidth=1.5,
            zorder=4,
        )
        ax_rel.scatter(
            idx,
            row["rel_cool"],
            s=64,
            facecolor=row["color"],
            edgecolor="white",
            linewidth=0.6,
            zorder=5,
        )
    ax_rel.axhline(0.99, color="#9B2226", linestyle=(0, (4, 3)), linewidth=1.2, zorder=1)
    ax_rel.text(len(df) - 0.35, 0.99035, "cooling gate = 0.99", fontsize=8.5, color="#9B2226", ha="right")
    ax_rel.set_ylim(0.985, 1.0015)
    ax_rel.set_yticks([0.985, 0.99, 0.995, 1.0])
    ax_rel.set_title("Reliability profile")
    ax_rel.set_ylabel("Reliability")
    ax_rel.set_xticks(x, labels, rotation=16, ha="right")
    _style_axes(ax_rel, grid_axis="y")
    rel_legend = [
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="white", markeredgecolor="#1F2430", markersize=7, linewidth=0, label="Heat"),
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="#1F2430", markeredgecolor="white", markersize=7, linewidth=0, label="Cooling"),
    ]
    ax_rel.legend(handles=rel_legend, loc="lower left", ncol=2)
    _panel_label(ax_rel, "(b)")

    unmet_c = df["unmet_c_mwh"].to_numpy()
    unmet_h = np.maximum(df["unmet_h_mwh"].to_numpy(), 0.02)
    ax_unmet.bar(
        x,
        unmet_c,
        color=[lighten_color(color, 0.18) for color in colors],
        edgecolor=colors,
        linewidth=1.1,
        width=0.68,
        zorder=3,
        label="Cooling unmet",
    )
    ax_unmet.bar(
        x,
        unmet_h,
        color="#D9DEE7",
        edgecolor="#B9C0CB",
        linewidth=0.8,
        width=0.68,
        zorder=4,
        label="Heat unmet",
    )
    for idx, value in enumerate(df["unmet_c_mwh"]):
        ax_unmet.text(idx, value + max(df["unmet_c_mwh"]) * 0.02 + 0.25, f"{value:.1f}", ha="center", va="bottom", fontsize=9, color="#1F2430")
    ax_unmet.set_title("Unmet energy")
    ax_unmet.set_ylabel("MWh")
    ax_unmet.set_xticks(x, labels, rotation=16, ha="right")
    _style_axes(ax_unmet, grid_axis="y")
    ax_unmet.legend(loc="upper left")
    _panel_label(ax_unmet, "(c)")

    burden_bars = ax_burden.bar(
        x,
        df["export_penalty_m"],
        color=[lighten_color(color, 0.28) for color in colors],
        edgecolor=colors,
        linewidth=1.0,
        width=0.62,
        zorder=3,
    )
    ax_burden.set_title("Export friction and switching burden")
    ax_burden.set_ylabel("Export penalty (M)")
    ax_burden.set_xticks(x, labels, rotation=16, ha="right")
    _style_axes(ax_burden, grid_axis="y")
    ax_burden_2 = ax_burden.twinx()
    ax_burden_2.plot(
        x,
        df["starts_gt"],
        color="#3A3F46",
        linewidth=2.0,
        marker="o",
        markersize=5.5,
        markerfacecolor="white",
        markeredgewidth=1.3,
        zorder=5,
    )
    ax_burden_2.set_ylabel("GT starts", color="#3A3F46")
    ax_burden_2.tick_params(axis="y", colors="#3A3F46")
    ax_burden_2.spines["top"].set_visible(False)
    ax_burden_2.spines["left"].set_visible(False)
    ax_burden_2.spines["right"].set_linewidth(0.8)
    ax_burden_2.spines["right"].set_color("#AAB2BF")
    burden_legend = [
        Patch(facecolor="#DDE7F0", edgecolor="#6C7A89", label="Export penalty"),
        Line2D([0], [0], color="#3A3F46", marker="o", markerfacecolor="white", label="GT starts"),
    ]
    ax_burden.legend(handles=burden_legend, loc="upper right")
    _panel_label(ax_burden, "(d)")

    fig.suptitle("Same-info DRL main results on the latest Kaggle MLP cohort", fontsize=16, fontweight="bold", color="#1F2430", y=0.95)
    return fig


def plot_convergence_overview(bundle: dict[str, Any]) -> plt.Figure:
    fig, axes = plt.subplots(2, 1, figsize=(14.6, 9.2), sharex=True)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.09, top=0.84, hspace=0.16)
    ax_cost, ax_cool = axes

    for algo in bundle["order"]:
        curve = bundle["curves"][algo]
        row = bundle["metrics"].set_index("algo").loc[algo]
        color = row["color"]
        label = row["label"]

        ax_cost.plot(curve["timesteps"], curve["mean_total_cost"] / 1_000_000.0, color=color, linewidth=2.05, label=label)
        ax_cool.plot(curve["timesteps"], curve["reliability_min__cooling"], color=color, linewidth=2.05, label=label)

        selected_t = float(row["selected_best_t"])
        reward_t = float(row["reward_best_t"])
        selected_idx = (curve["timesteps"] - selected_t).abs().idxmin()
        reward_idx = (curve["timesteps"] - reward_t).abs().idxmin()

        ax_cost.scatter(curve.loc[selected_idx, "timesteps"], curve.loc[selected_idx, "mean_total_cost"] / 1_000_000.0, color=color, s=52, zorder=4)
        ax_cool.scatter(curve.loc[selected_idx, "timesteps"], curve.loc[selected_idx, "reliability_min__cooling"], color=color, s=52, zorder=4)

        ax_cost.scatter(
            curve.loc[reward_idx, "timesteps"],
            curve.loc[reward_idx, "mean_total_cost"] / 1_000_000.0,
            facecolor="white",
            edgecolor=color,
            linewidth=1.4,
            s=56,
            zorder=5,
        )
        ax_cool.scatter(
            curve.loc[reward_idx, "timesteps"],
            curve.loc[reward_idx, "reliability_min__cooling"],
            facecolor="white",
            edgecolor=color,
            linewidth=1.4,
            s=56,
            zorder=5,
        )

        low_lr_t = row["low_lr_event_t"]
        if low_lr_t is not None and not np.isnan(low_lr_t):
            event_idx = (curve["timesteps"] - float(low_lr_t)).abs().idxmin()
            ax_cost.scatter(
                curve.loc[event_idx, "timesteps"],
                curve.loc[event_idx, "mean_total_cost"] / 1_000_000.0,
                marker="D",
                color=color,
                edgecolor="white",
                linewidth=0.8,
                s=42,
                zorder=6,
            )
            ax_cool.scatter(
                curve.loc[event_idx, "timesteps"],
                curve.loc[event_idx, "reliability_min__cooling"],
                marker="D",
                color=color,
                edgecolor="white",
                linewidth=0.8,
                s=42,
                zorder=6,
            )
            ax_cost.axvline(float(low_lr_t), color=color, linestyle=(0, (2, 3)), linewidth=1.0, zorder=1)
            ax_cool.axvline(float(low_lr_t), color=color, linestyle=(0, (2, 3)), linewidth=1.0, zorder=1)

    ax_cost.set_title("Training-eval cost trajectory", pad=10)
    ax_cost.set_ylabel("Mean total cost (M)")
    _style_axes(ax_cost, grid_axis="both")
    ax_cost.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}"))
    _panel_label(ax_cost, "(a)")

    ax_cool.axhline(0.99, color="#9B2226", linestyle=(0, (4, 3)), linewidth=1.2)
    ax_cool.text(2_000_000, 0.99035, "cooling gate = 0.99", color="#9B2226", fontsize=8.5, ha="right")
    ax_cool.set_title("Cooling reliability trajectory on the fixed training eval pool", pad=10)
    ax_cool.set_ylabel("Min cooling reliability")
    ax_cool.set_xlabel("Timesteps")
    ax_cool.set_ylim(0.78, 1.01)
    ax_cool.xaxis.set_major_formatter(FuncFormatter(_format_million_steps))
    _style_axes(ax_cool, grid_axis="both")
    _panel_label(ax_cool, "(b)")

    algo_handles = [
        Line2D([0], [0], color=row["color"], lw=2.2, label=row["label"])
        for _, row in bundle["metrics"].iterrows()
    ]
    marker_handles = [
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="#1F2430", markeredgecolor="#1F2430", markersize=6.5, label="Selected best"),
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="white", markeredgecolor="#1F2430", markersize=6.5, label="Reward best"),
        Line2D([0], [0], marker="D", linestyle="None", markerfacecolor="#1F2430", markeredgecolor="white", markersize=6.0, label="Low-LR switch"),
        Line2D([0], [0], color="#6C7480", linestyle=(0, (2, 3)), linewidth=1.0, label="Low-LR timing"),
    ]
    fig.legend(
        algo_handles,
        [handle.get_label() for handle in algo_handles],
        loc="upper left",
        ncol=3,
        bbox_to_anchor=(0.075, 0.985),
        handlelength=2.8,
        columnspacing=1.5,
    )
    fig.legend(
        marker_handles,
        [handle.get_label() for handle in marker_handles],
        loc="upper right",
        ncol=4,
        bbox_to_anchor=(0.985, 0.985),
        handlelength=1.8,
        columnspacing=1.2,
    )
    fig.suptitle("Convergence overview of the latest Kaggle MLP cohort", fontsize=16, fontweight="bold", color="#1F2430", y=0.965)
    fig.text(
        0.075,
        0.875,
        "Lines show the fixed-pool evaluation trajectory. Filled circles mark the selected checkpoint, hollow circles mark the reward-best point, and diamonds show the low-LR transition.",
        fontsize=9.2,
        color="#5B6574",
        ha="left",
    )
    return fig


def plot_behavior_diagnostics(bundle: dict[str, Any]) -> plt.Figure:
    df = bundle["metrics"].copy()
    labels = df["label"].tolist()
    colors = df["color"].tolist()
    x = np.arange(len(df))

    fig, axes = plt.subplots(2, 2, figsize=(14.4, 9.2))
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.12, top=0.86, wspace=0.15, hspace=0.28)
    ax_unmet, ax_export, ax_starts, ax_toggle = axes.flat

    ax_unmet.bar(x, df["cool_unmet_steps"], color=[lighten_color(c, 0.18) for c in colors], edgecolor=colors, linewidth=1.0, width=0.66)
    ax_unmet.set_title("Cooling shortage frequency")
    ax_unmet.set_ylabel("cool_unmet_steps")
    ax_unmet.set_xticks(x, labels, rotation=16, ha="right")
    _style_axes(ax_unmet, grid_axis="y")
    _panel_label(ax_unmet, "(a)")

    ax_export.bar(x, df["export_over_soft_cap_steps"], color=[lighten_color(c, 0.3) for c in colors], edgecolor=colors, linewidth=1.0, width=0.66)
    ax_export.set_title("Export soft-cap pressure")
    ax_export.set_ylabel("export_over_soft_cap_steps")
    ax_export.set_xticks(x, labels, rotation=16, ha="right")
    _style_axes(ax_export, grid_axis="y")
    _panel_label(ax_export, "(b)")

    width = 0.32
    ax_starts.bar(x - width / 2, df["starts_gt"], width=width, color="#CDD9E5", edgecolor="#7A8797", linewidth=0.9, label="GT starts")
    ax_starts.bar(x + width / 2, df["starts_ech"], width=width, color="#E9D8C3", edgecolor="#B07D3C", linewidth=0.9, label="ECH starts")
    ax_starts.set_title("Unit start burden")
    ax_starts.set_ylabel("starts")
    ax_starts.set_xticks(x, labels, rotation=16, ha="right")
    ax_starts.legend(loc="upper left")
    _style_axes(ax_starts, grid_axis="y")
    _panel_label(ax_starts, "(c)")

    ax_toggle.bar(x, df["gt_toggle_steps"], color=[lighten_color(c, 0.12) for c in colors], edgecolor=colors, linewidth=1.0, width=0.66)
    ax_toggle.set_title("GT toggle burden")
    ax_toggle.set_ylabel("gt_toggle_steps")
    ax_toggle.set_xticks(x, labels, rotation=16, ha="right")
    _style_axes(ax_toggle, grid_axis="y")
    _panel_label(ax_toggle, "(d)")

    fig.suptitle("Supplementary behavior diagnostics", fontsize=15.5, fontweight="bold", color="#1F2430", y=0.95)
    return fig


def plot_physics_uptake(bundle: dict[str, Any]) -> plt.Figure:
    df = bundle["metrics"].copy()
    labels = df["label"].tolist()
    colors = df["color"].tolist()
    x = np.arange(len(df))

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.9))
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.17, top=0.82, wspace=0.14)
    ax_blocked, ax_phys, ax_qabs = axes

    ax_blocked.bar(x, df["abs_blocked_rate"], color=[lighten_color(c, 0.22) for c in colors], edgecolor=colors, linewidth=1.0, width=0.64)
    ax_blocked.set_title("ABS blocked rate")
    ax_blocked.set_ylabel("Fraction of steps")
    ax_blocked.set_xticks(x, labels, rotation=16, ha="right")
    ax_blocked.set_ylim(0, 1.0)
    _style_axes(ax_blocked, grid_axis="y")
    _panel_label(ax_blocked, "(a)")

    ax_phys.bar(x, df["u_abs_phys_rate"], color=[lighten_color(c, 0.08) for c in colors], edgecolor=colors, linewidth=1.0, width=0.64)
    ax_phys.set_title("Physical ABS activation")
    ax_phys.set_ylabel("u_abs_phys nonzero rate")
    ax_phys.set_xticks(x, labels, rotation=16, ha="right")
    ax_phys.set_ylim(0, 1.0)
    _style_axes(ax_phys, grid_axis="y")
    _panel_label(ax_phys, "(b)")

    ax_qabs.bar(x, df["q_abs_mean_mw"], color=[lighten_color(c, 0.16) for c in colors], edgecolor=colors, linewidth=1.0, width=0.64)
    ax_qabs.set_title("Delivered ABS cooling")
    ax_qabs.set_ylabel("Mean q_abs_cool (MW)")
    ax_qabs.set_xticks(x, labels, rotation=16, ha="right")
    _style_axes(ax_qabs, grid_axis="y")
    _panel_label(ax_qabs, "(c)")

    fig.suptitle("Supplementary physics-uptake evidence", fontsize=15.5, fontweight="bold", color="#1F2430", y=0.95)
    return fig


def main() -> None:
    _set_theme()
    bundle = _load_dataset()

    figures = {
        "latest_kaggle_mlp_main_results": plot_main_results(bundle),
        "latest_kaggle_mlp_convergence_overview": plot_convergence_overview(bundle),
        "latest_kaggle_mlp_behavior_diagnostics": plot_behavior_diagnostics(bundle),
        "latest_kaggle_mlp_physics_uptake": plot_physics_uptake(bundle),
    }
    for stem, fig in figures.items():
        _save_figure(fig, stem)
        plt.close(fig)


if __name__ == "__main__":
    main()
