from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_ROOT = ROOT / "results"
DEFAULT_ARCHIVE_TAG = "paper_ready_2026-03-31"
DEFAULT_FULL_TABLE = ROOT / "results" / "tables" / "paper" / "drl_all_available_multi_seed_yearly_eval_2026-03-31.csv"
DEFAULT_AGG_TABLE = ROOT / "results" / "tables" / "paper" / "drl_all_available_multi_seed_yearly_eval_aggregate_2026-03-31.csv"

MODEL_ORDER = [
    "DDPG+rule_residual",
    "SAC+rule_residual",
    "TD3+rule_residual",
    "rbDQN",
    "PPO+rule_residual",
]
MODEL_COLORS = {
    "DDPG+rule_residual": "#416A52",
    "SAC+rule_residual": "#6A7E8F",
    "TD3+rule_residual": "#8B7A4A",
    "rbDQN": "#1C617A",
    "PPO+rule_residual": "#B46E42",
}
MODEL_SHORT = {
    "DDPG+rule_residual": "DDPG",
    "SAC+rule_residual": "SAC",
    "TD3+rule_residual": "TD3",
    "rbDQN": "rbDQN",
    "PPO+rule_residual": "PPO",
}
CASE_MODELS = ["rbDQN", "DDPG+rule_residual", "PPO+rule_residual"]
CASE_SIGNAL_COLORS = {
    "demand": "#2B2B2B",
    "abs": "#1C617A",
    "ech": "#B46E42",
    "gt": "#416A52",
    "grid": "#6A7E8F",
    "tes": "#B38B59",
}
CASE_STRATEGY_NOTES = {
    "rbDQN": "Steadier ABS-TES buffering",
    "DDPG+rule_residual": "GT-led lower-cost dispatch",
    "PPO+rule_residual": "ECH-biased cooling support",
}


def _hex_to_rgb(color: str) -> tuple[float, float, float]:
    value = color.lstrip("#")
    return tuple(int(value[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def lighten_color(color: str, amount: float = 0.25) -> tuple[float, float, float]:
    r, g, b = _hex_to_rgb(color)
    return tuple(1.0 - (1.0 - channel) * (1.0 - amount) for channel in (r, g, b))


def _set_theme() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIX Two Text", "STIXGeneral", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "figure.facecolor": "#F6F1E8",
            "axes.facecolor": "#FBF8F2",
            "savefig.facecolor": "#F6F1E8",
            "savefig.bbox": "tight",
            "axes.edgecolor": "#CDBFAE",
            "axes.labelcolor": "#1D232A",
            "axes.titlesize": 13.5,
            "axes.titleweight": "semibold",
            "axes.labelsize": 10.8,
            "axes.linewidth": 0.9,
            "xtick.color": "#1D232A",
            "ytick.color": "#1D232A",
            "xtick.labelsize": 9.6,
            "ytick.labelsize": 9.6,
            "grid.color": "#E6DDCF",
            "grid.linewidth": 0.8,
            "grid.alpha": 1.0,
            "legend.frameon": False,
            "legend.fontsize": 8.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _style_axes(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    ax.grid(True, axis=grid_axis, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.spines["left"].set_color("#CDBFAE")
    ax.spines["bottom"].set_color("#CDBFAE")
    ax.tick_params(length=3.3, width=0.8)


def _category_backdrop(ax: plt.Axes, positions: np.ndarray) -> None:
    for idx, model in enumerate(MODEL_ORDER):
        color = MODEL_COLORS[model]
        ax.axvspan(
            positions[idx] - 0.42,
            positions[idx] + 0.42,
            color=lighten_color(color, 0.82),
            alpha=0.22,
            zorder=0,
        )


def _label_box(ax: plt.Axes, x: float, y: float, text: str, color: str) -> None:
    ax.text(
        x,
        y,
        text,
        color=color,
        fontsize=8.8,
        ha="left",
        va="center",
        bbox={
            "boxstyle": "round,pad=0.22,rounding_size=0.12",
            "fc": lighten_color(color, 0.82),
            "ec": "none",
            "alpha": 0.95,
        },
        zorder=6,
    )


def _format_million_steps(value: float, _: float) -> str:
    return f"{value / 1_000_000.0:.1f}M"


def _figure_dirs(results_root: Path) -> dict[str, Path]:
    base = results_root / "figures" / "paper" / "multi_seed_nature"
    return {
        "png": base / "png",
        "pdf": base / "pdf",
        "eps": base / "eps",
        "tiff": base / "tiff",
    }


def _table_dir(results_root: Path) -> Path:
    return results_root / "tables" / "paper" / "multi_seed_nature"


def _paper_figure_dir() -> Path:
    return ROOT / "cchp-paper" / "figures" / "generated"


def _save_figure(fig: plt.Figure, stem: str, figure_dirs: dict[str, Path]) -> None:
    for fmt, out_dir in figure_dirs.items():
        out_dir.mkdir(parents=True, exist_ok=True)
        dpi = 600 if fmt in {"png", "tiff"} else 300
        output_path = out_dir / f"{stem}.{fmt}"
        fig.savefig(output_path, format=fmt, dpi=dpi)
        if fmt == "pdf":
            paper_dir = _paper_figure_dir()
            paper_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(output_path, paper_dir / output_path.name)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_learning_curve(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "timesteps" not in df.columns:
        return None
    return df.sort_values("timesteps").reset_index(drop=True)


def _load_bundle(results_root: Path, *, archive_tag: str, full_table: Path, agg_table: Path) -> dict[str, Any]:
    full_df = pd.read_csv(full_table)
    agg_df = pd.read_csv(agg_table)
    archive_root = results_root / "archive" / archive_tag
    baseline_catalog = pd.read_csv(archive_root / "catalogs" / "baseline_catalog.csv")

    full_df["color"] = full_df["model"].map(MODEL_COLORS)
    full_df["short_label"] = full_df["model"].map(MODEL_SHORT)
    agg_df["color"] = agg_df["model"].map(MODEL_COLORS)
    agg_df["short_label"] = agg_df["model"].map(MODEL_SHORT)

    curves: dict[str, list[pd.DataFrame]] = {model: [] for model in MODEL_ORDER}
    curve_manifest_rows: list[dict[str, Any]] = []
    catalog_df = pd.read_csv(archive_root / "catalogs" / "drl_eval_catalog.csv")
    for _, row in catalog_df.iterrows():
        model = str(row["model"])
        archive_dir = Path(str(row["archive_dir"]))
        curve_path = archive_dir / "train" / "learning_curve_eval.csv"
        curve = _load_learning_curve(curve_path)
        if curve is None:
            continue
        curve["model"] = model
        curve["train_seed"] = int(row["train_seed"])
        curves[model].append(curve)
        curve_manifest_rows.append(
            {
                "model": model,
                "train_seed": int(row["train_seed"]),
                "curve_path": str(curve_path),
                "points": int(len(curve)),
                "max_timesteps": float(curve["timesteps"].max()),
            }
        )

    curve_manifest = pd.DataFrame(curve_manifest_rows)
    allowed_pairs = {
        (str(row["model"]), int(row["train_seed"]))
        for _, row in full_df.iterrows()
    }
    curve_manifest = curve_manifest[
        curve_manifest.apply(lambda row: (str(row["model"]), int(row["train_seed"])) in allowed_pairs, axis=1)
    ].reset_index(drop=True)
    filtered_curves: dict[str, list[pd.DataFrame]] = {model: [] for model in MODEL_ORDER}
    for model, model_curves in curves.items():
        for curve in model_curves:
            if (model, int(curve["train_seed"].iloc[0])) in allowed_pairs:
                filtered_curves[model].append(curve)
    curves = filtered_curves
    baseline_rows: list[dict[str, Any]] = []
    for _, row in baseline_catalog.iterrows():
        archive_dir = Path(str(row["archive_dir"]))
        summary = _load_json(archive_dir / "eval" / "summary.json")
        baseline_rows.append(
            {
                "slug": str(row["slug"]),
                "label": str(row["label"]),
                "total_cost_m": float(summary["total_cost"]) / 1_000_000.0,
                "rel_heat": float(summary["reliability"]["heat"]),
                "rel_cool": float(summary["reliability"]["cooling"]),
                "violation_rate": float(summary["violation_rate"]),
                "starts_gt": int(summary["starts"]["gt"]),
                "starts_ech": int(summary["starts"]["ech"]),
                "archive_dir": str(archive_dir),
            }
        )
    baselines_df = pd.DataFrame(baseline_rows)

    return {
        "full": full_df,
        "aggregate": agg_df,
        "baselines": baselines_df,
        "curves": curves,
        "curve_manifest": curve_manifest,
        "catalog": catalog_df,
        "archive_root": archive_root,
    }


def _curve_band(curves: list[pd.DataFrame], value_column: str) -> pd.DataFrame:
    if not curves:
        return pd.DataFrame(columns=["timesteps", "mean", "std", "n"])

    joined: pd.DataFrame | None = None
    for idx, curve in enumerate(curves):
        current = curve[["timesteps", value_column]].rename(columns={value_column: f"seed_{idx}"})
        joined = current if joined is None else joined.merge(current, on="timesteps", how="outer")
    assert joined is not None
    joined = joined.sort_values("timesteps").reset_index(drop=True)
    value_cols = [col for col in joined.columns if col.startswith("seed_")]
    joined["mean"] = joined[value_cols].mean(axis=1, skipna=True)
    joined["std"] = joined[value_cols].std(axis=1, skipna=True).fillna(0.0)
    joined["n"] = joined[value_cols].count(axis=1)
    return joined[["timesteps", "mean", "std", "n"]]


def _cooling_column(curve: pd.DataFrame) -> str:
    for candidate in ("reliability_mean__cooling", "reliability_mean_cooling"):
        if candidate in curve.columns:
            return candidate
    raise KeyError("Cooling reliability column not found in learning_curve_eval.csv")


def _export_plot_tables(bundle: dict[str, Any], results_root: Path) -> dict[str, pd.DataFrame]:
    out_dir = _table_dir(results_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    frontier = bundle["aggregate"].copy()
    bundle["full"].to_csv(out_dir / "multi_seed_full_runs.csv", index=False, encoding="utf-8")
    frontier.to_csv(out_dir / "multi_seed_frontier_data.csv", index=False, encoding="utf-8")
    bundle["baselines"].to_csv(out_dir / "baseline_reference_data.csv", index=False, encoding="utf-8")
    bundle["curve_manifest"].to_csv(out_dir / "learning_curve_manifest.csv", index=False, encoding="utf-8")

    cost_bands: list[pd.DataFrame] = []
    cooling_bands: list[pd.DataFrame] = []
    for model in MODEL_ORDER:
        curves = bundle["curves"].get(model, [])
        if not curves:
            continue
        cost_band = _curve_band(curves, "mean_total_cost")
        cost_band["model"] = model
        cost_bands.append(cost_band)

        cool_band = _curve_band(curves, _cooling_column(curves[0]))
        cool_band["model"] = model
        cooling_bands.append(cool_band)

    cost_df = pd.concat(cost_bands, ignore_index=True) if cost_bands else pd.DataFrame()
    cooling_df = pd.concat(cooling_bands, ignore_index=True) if cooling_bands else pd.DataFrame()
    if not cost_df.empty:
        cost_df.to_csv(out_dir / "training_cost_bands.csv", index=False, encoding="utf-8")
    if not cooling_df.empty:
        cooling_df.to_csv(out_dir / "training_cooling_bands.csv", index=False, encoding="utf-8")

    manifest = {
        "archive_root": str(bundle["archive_root"]),
        "full_table_rows": int(len(bundle["full"])),
        "aggregate_rows": int(len(bundle["aggregate"])),
        "figures": [
            "multi_seed_frontier",
            "multi_seed_cost_strip",
            "multi_seed_training_cost",
            "multi_seed_training_cooling",
            "multi_seed_cooling_strip",
            "multi_seed_case_window",
        ],
    }
    (out_dir / "multi_seed_nature_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"cost_bands": cost_df, "cooling_bands": cooling_df}


def _pick_case_representatives(bundle: dict[str, Any]) -> pd.DataFrame:
    catalog = bundle["catalog"].copy()
    full_df = bundle["full"].copy()
    rows: list[pd.Series] = []
    for model in CASE_MODELS:
        subset = catalog[catalog["model"] == model].copy()
        rep = subset[subset["representative_run"] == True]
        if not rep.empty:
            rows.append(rep.iloc[0])
            continue
        full_subset = full_df[full_df["model"] == model].copy()
        good = full_subset[full_subset["rel_cool"] >= 0.99].copy()
        if good.empty:
            good = full_subset
        chosen = good.sort_values(["total_cost_m", "train_seed"]).iloc[0]
        matched = subset[subset["train_seed"] == chosen["train_seed"]]
        rows.append(matched.iloc[0] if not matched.empty else chosen)
    return pd.DataFrame(rows).reset_index(drop=True)


def _load_case_step_log(archive_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(archive_dir / "eval" / "step_log_light.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def _select_case_window(reference_log: pd.DataFrame, window_steps: int = 96) -> tuple[int, int]:
    demand_c_mw = reference_log["energy_demand_c_mwh"].astype(float) * 4.0
    rolling = demand_c_mw.rolling(window_steps, min_periods=window_steps).mean()
    end_idx = int(rolling.idxmax())
    start_idx = max(0, end_idx - window_steps + 1)
    return start_idx, start_idx + window_steps


def _extract_case_series(step_log: pd.DataFrame, start_idx: int, end_idx: int) -> pd.DataFrame:
    window = step_log.iloc[start_idx:end_idx].copy().reset_index(drop=True)
    window["cooling_demand_mw"] = window["energy_demand_c_mwh"].astype(float) * 4.0
    window["grid_net_mw"] = window["p_grid_import_mw"].astype(float) - window["p_grid_export_mw"].astype(float)
    window["gt_power_mw"] = window["p_gt_mw"].astype(float)
    window["abs_cooling_mw"] = window["q_abs_cool_mw"].astype(float)
    window["ech_cooling_mw"] = window["q_ech_cool_mw"].astype(float)
    window["tes_energy_mwh"] = window["e_tes_mwh"].astype(float)
    return window[
        [
            "timestamp",
            "cooling_demand_mw",
            "abs_cooling_mw",
            "ech_cooling_mw",
            "gt_power_mw",
            "grid_net_mw",
            "tes_energy_mwh",
        ]
    ]


def _export_case_window_table(bundle: dict[str, Any], results_root: Path) -> pd.DataFrame:
    reps = _pick_case_representatives(bundle)
    ref_log = _load_case_step_log(Path(str(reps.iloc[0]["archive_dir"])))
    start_idx, end_idx = _select_case_window(ref_log)
    rows: list[pd.DataFrame] = []
    for _, rep in reps.iterrows():
        step_log = _load_case_step_log(Path(str(rep["archive_dir"])))
        series = _extract_case_series(step_log, start_idx, end_idx)
        series["model"] = str(rep["model"])
        series["short_label"] = MODEL_SHORT[str(rep["model"])]
        series["train_seed"] = int(rep["train_seed"])
        rows.append(series)
    out = pd.concat(rows, ignore_index=True)
    out_dir = _table_dir(results_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "case_window_profiles.csv", index=False, encoding="utf-8")
    return out


def _case_window_metrics(window: pd.DataFrame) -> dict[str, float]:
    demand = window["cooling_demand_mw"].clip(lower=0.0)
    supplied = window["abs_cooling_mw"].clip(lower=0.0) + window["ech_cooling_mw"].clip(lower=0.0)
    supplied_sum = float(supplied.sum()) or 1.0
    abs_share = float(window["abs_cooling_mw"].clip(lower=0.0).sum() / supplied_sum)
    ech_share = float(window["ech_cooling_mw"].clip(lower=0.0).sum() / supplied_sum)
    gt_mean = float(window["gt_power_mw"].mean())
    grid_import_mean = float(window["grid_net_mw"].clip(lower=0.0).mean())
    tes_swing = float(window["tes_energy_mwh"].max() - window["tes_energy_mwh"].min())
    grid_ramp_std = float(window["grid_net_mw"].diff().fillna(0.0).std(ddof=0))
    return {
        "abs_share": abs_share,
        "ech_share": ech_share,
        "gt_mean": gt_mean,
        "grid_import_mean": grid_import_mean,
        "tes_swing": tes_swing,
        "grid_ramp_std": grid_ramp_std,
    }


def _case_summary_text(model: str, metrics: dict[str, float]) -> str:
    if model == "rbDQN":
        return (
            f"ABS share {metrics['abs_share']:.0%} | "
            f"grid ramp std {metrics['grid_ramp_std']:.1f} MW\n"
            f"TES swing {metrics['tes_swing']:.1f} MWh"
        )
    if model == "DDPG+rule_residual":
        return (
            f"GT mean {metrics['gt_mean']:.1f} MW | "
            f"grid import {metrics['grid_import_mean']:.1f} MW\n"
            f"ECH share {metrics['ech_share']:.0%}"
        )
    return (
        f"ECH share {metrics['ech_share']:.0%} | "
        f"ABS share {metrics['abs_share']:.0%}\n"
        f"GT mean {metrics['gt_mean']:.1f} MW"
    )


def plot_multi_seed_frontier(bundle: dict[str, Any]) -> plt.Figure:
    df = bundle["aggregate"].copy()
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    offsets = {
        "DDPG+rule_residual": (0.00018, -0.06),
        "SAC+rule_residual": (0.00018, 0.09),
        "TD3+rule_residual": (0.00018, 0.02),
        "rbDQN": (0.00018, 0.02),
        "PPO+rule_residual": (0.00018, 0.08),
    }

    ax.axvspan(0.978, 0.99, color="#F6E6E1", alpha=0.88, zorder=0)
    ax.axvspan(0.99, 1.0007, color="#EAF2EC", alpha=0.95, zorder=0)
    ax.axvline(0.99, color="#AF3D35", linestyle=(0, (4, 3)), linewidth=1.1, zorder=1)
    ax.text(0.9784, df["total_cost_mean"].max() + 0.11, "below gate", color="#A25D53", fontsize=8.2, ha="left")
    ax.text(0.9902, df["total_cost_mean"].max() + 0.11, "engineering-relevant region", color="#416A52", fontsize=8.2, ha="left")

    for _, row in df.iterrows():
        color = row["color"]
        ax.errorbar(
            row["rel_cool_mean"],
            row["total_cost_mean"],
            xerr=row["rel_cool_std"],
            yerr=row["total_cost_std"],
            fmt="o",
            ms=9.2,
            color=color,
            mfc=lighten_color(color, 0.05),
            mec=color,
            elinewidth=1.2,
            capsize=3.0,
            zorder=4,
        )
        _label_box(
            ax,
            row["rel_cool_mean"] + offsets[row["model"]][0],
            row["total_cost_mean"] + offsets[row["model"]][1],
            f"{row['short_label']} (n={int(row['n_seeds'])})",
            color,
        )

    baseline_colors = {
        "Oracle MILP (strict, h32)": "#222222",
        "Rule baseline (h16)": "#8A8F97",
    }
    for _, row in bundle["baselines"].iterrows():
        color = baseline_colors.get(str(row["label"]), "#555555")
        marker = "D" if "Oracle" in str(row["label"]) else "s"
        ax.scatter(
            row["rel_cool"],
            row["total_cost_m"],
            s=62,
            marker=marker,
            color="#FBF8F2",
            edgecolor=color,
            linewidth=1.2,
            zorder=5,
        )
        _label_box(
            ax,
            row["rel_cool"] + 0.00018,
            row["total_cost_m"] - 0.04,
            str(row["label"]),
            color,
        )

    ax.set_xlabel("Cooling reliability")
    ax.set_ylabel("Annual total cost (million CNY-equivalent)")
    ax.set_title("Multi-seed economic-reliability frontier")
    ax.set_xlim(0.978, 1.0007)
    _style_axes(ax, grid_axis="both")
    return fig


def plot_multi_seed_cost_strip(bundle: dict[str, Any]) -> plt.Figure:
    df = bundle["full"].copy()
    fig, ax = plt.subplots(figsize=(7.5, 4.9))
    positions = np.arange(len(MODEL_ORDER))
    rng = np.random.default_rng(20260331)
    _category_backdrop(ax, positions)

    for idx, model in enumerate(MODEL_ORDER):
        subset = df[df["model"] == model].copy()
        if subset.empty:
            continue
        violin = ax.violinplot(
            subset["total_cost_m"],
            positions=[positions[idx]],
            widths=0.70,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for body in violin["bodies"]:
            body.set_facecolor(lighten_color(MODEL_COLORS[model], 0.76))
            body.set_edgecolor(MODEL_COLORS[model])
            body.set_alpha(0.26)
            body.set_linewidth(0.9)
        jitter = rng.uniform(-0.16, 0.16, size=len(subset))
        color = MODEL_COLORS[model]
        ax.scatter(
            np.full(len(subset), positions[idx]) + jitter,
            subset["total_cost_m"],
            s=38,
            color=lighten_color(color, 0.14),
            edgecolor=color,
            linewidth=0.9,
            alpha=0.95,
            zorder=3,
        )
        mean_value = subset["total_cost_m"].mean()
        std_value = subset["total_cost_m"].std(ddof=1) if len(subset) > 1 else 0.0
        ax.errorbar(
            positions[idx],
            mean_value,
            yerr=std_value,
            fmt="D",
            ms=6.5,
            color=color,
            mfc="#FBF8F2",
            mec=color,
            elinewidth=1.2,
            capsize=3.0,
            zorder=5,
        )

    ax.set_xticks(positions, [MODEL_SHORT[model] for model in MODEL_ORDER])
    ax.set_ylabel("Annual total cost (million CNY-equivalent)")
    ax.set_title("Seed sensitivity of yearly economic performance")
    _style_axes(ax, grid_axis="y")
    return fig


def plot_training_cost(bundle: dict[str, Any]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    for model in MODEL_ORDER:
        curves = bundle["curves"].get(model, [])
        if not curves:
            continue
        band = _curve_band(curves, "mean_total_cost")
        color = MODEL_COLORS[model]
        ax.plot(band["timesteps"], band["mean"] / 1_000_000.0, color=color, linewidth=2.0, label=MODEL_SHORT[model])
        stable = band["n"] >= 3
        ax.fill_between(
            band.loc[stable, "timesteps"],
            (band.loc[stable, "mean"] - band.loc[stable, "std"]) / 1_000_000.0,
            (band.loc[stable, "mean"] + band.loc[stable, "std"]) / 1_000_000.0,
            color=lighten_color(color, 0.45),
            alpha=0.16,
        )
        ax.scatter(
            band["timesteps"].iloc[-1],
            band["mean"].iloc[-1] / 1_000_000.0,
            s=22,
            color=color,
            zorder=5,
        )

    ax.set_xlabel("Training timesteps")
    ax.set_ylabel("Windowed evaluation cost (million)")
    ax.set_title("Training robustness in evaluation cost")
    ax.xaxis.set_major_formatter(FuncFormatter(_format_million_steps))
    ax.set_xlim(left=0.0)
    _style_axes(ax, grid_axis="y")
    ax.legend(ncol=3, loc="upper right")
    return fig


def plot_training_cooling(bundle: dict[str, Any]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.axhspan(0.99, 1.0015, color="#EAF2EC", alpha=0.95, zorder=0)

    for model in MODEL_ORDER:
        curves = bundle["curves"].get(model, [])
        if not curves:
            continue
        band = _curve_band(curves, _cooling_column(curves[0]))
        color = MODEL_COLORS[model]
        ax.plot(band["timesteps"], band["mean"], color=color, linewidth=2.0, label=MODEL_SHORT[model])
        stable = band["n"] >= 3
        ax.fill_between(
            band.loc[stable, "timesteps"],
            band.loc[stable, "mean"] - band.loc[stable, "std"],
            band.loc[stable, "mean"] + band.loc[stable, "std"],
            color=lighten_color(color, 0.45),
            alpha=0.16,
        )
        ax.scatter(
            band["timesteps"].iloc[-1],
            band["mean"].iloc[-1],
            s=22,
            color=color,
            zorder=5,
        )

    ax.axhline(0.99, color="#AF3D35", linestyle=(0, (4, 3)), linewidth=1.05)
    ax.text(0.012, 0.99055, "cooling gate", transform=ax.get_yaxis_transform(), color="#AF3D35", fontsize=8.5)
    ax.set_xlabel("Training timesteps")
    ax.set_ylabel("Mean cooling reliability")
    ax.set_title("Training robustness in cooling adequacy")
    ax.xaxis.set_major_formatter(FuncFormatter(_format_million_steps))
    ax.set_xlim(left=0.0)
    ax.set_ylim(0.985, 1.0015)
    _style_axes(ax, grid_axis="y")
    ax.legend(ncol=3, loc="lower right")
    return fig


def plot_multi_seed_cooling_strip(bundle: dict[str, Any]) -> plt.Figure:
    df = bundle["full"].copy()
    fig, ax = plt.subplots(figsize=(7.5, 4.9))
    positions = np.arange(len(MODEL_ORDER))
    rng = np.random.default_rng(20260401)
    _category_backdrop(ax, positions)

    for idx, model in enumerate(MODEL_ORDER):
        subset = df[df["model"] == model].copy()
        if subset.empty:
            continue
        violin = ax.violinplot(
            subset["rel_cool"],
            positions=[positions[idx]],
            widths=0.70,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for body in violin["bodies"]:
            body.set_facecolor(lighten_color(MODEL_COLORS[model], 0.76))
            body.set_edgecolor(MODEL_COLORS[model])
            body.set_alpha(0.26)
            body.set_linewidth(0.9)
        jitter = rng.uniform(-0.16, 0.16, size=len(subset))
        color = MODEL_COLORS[model]
        ax.scatter(
            np.full(len(subset), positions[idx]) + jitter,
            subset["rel_cool"],
            s=38,
            color=lighten_color(color, 0.14),
            edgecolor=color,
            linewidth=0.9,
            alpha=0.95,
            zorder=3,
        )
        mean_value = subset["rel_cool"].mean()
        std_value = subset["rel_cool"].std(ddof=1) if len(subset) > 1 else 0.0
        ax.errorbar(
            positions[idx],
            mean_value,
            yerr=std_value,
            fmt="D",
            ms=6.5,
            color=color,
            mfc="#FBF8F2",
            mec=color,
            elinewidth=1.2,
            capsize=3.0,
            zorder=5,
        )

    ax.axhline(0.99, color="#AF3D35", linestyle=(0, (4, 3)), linewidth=1.05)
    ax.set_xticks(positions, [MODEL_SHORT[model] for model in MODEL_ORDER])
    ax.set_ylabel("Yearly cooling reliability")
    ax.set_title("Seed sensitivity of cooling adequacy")
    ax.set_ylim(0.978, 1.0012)
    _style_axes(ax, grid_axis="y")
    return fig


def plot_case_window(bundle: dict[str, Any], results_root: Path) -> plt.Figure:
    case_df = _export_case_window_table(bundle, results_root)
    reps = (
        case_df[["model", "short_label", "train_seed"]]
        .drop_duplicates()
        .set_index("model")
        .loc[CASE_MODELS]
        .reset_index()
    )
    fig, axes = plt.subplots(2, 3, figsize=(11.8, 6.2), sharex="col")
    fig.subplots_adjust(left=0.07, right=0.96, bottom=0.16, top=0.84, wspace=0.23, hspace=0.18)

    for col_idx, model in enumerate(CASE_MODELS):
        subset = case_df[case_df["model"] == model].copy().reset_index(drop=True)
        metrics = _case_window_metrics(subset)
        top_ax = axes[0, col_idx]
        bottom_ax = axes[1, col_idx]
        storage_ax = bottom_ax.twinx()
        x = np.arange(len(subset))

        top_ax.fill_between(
            x,
            0.0,
            subset["cooling_demand_mw"],
            color=lighten_color(CASE_SIGNAL_COLORS["demand"], 0.82),
            alpha=0.28,
            zorder=1,
        )
        top_ax.plot(
            x,
            subset["cooling_demand_mw"],
            color=CASE_SIGNAL_COLORS["demand"],
            linewidth=1.8,
            linestyle=(0, (4, 2)),
            label="Cooling demand",
        )
        top_ax.plot(
            x,
            subset["abs_cooling_mw"],
            color=CASE_SIGNAL_COLORS["abs"],
            linewidth=2.0,
            label="ABS cooling",
        )
        top_ax.plot(
            x,
            subset["ech_cooling_mw"],
            color=CASE_SIGNAL_COLORS["ech"],
            linewidth=2.0,
            label="ECH cooling",
        )
        _style_axes(top_ax, grid_axis="y")
        top_ax.set_title(f"{MODEL_SHORT[model]} (seed {int(reps.iloc[col_idx]['train_seed'])})", color=MODEL_COLORS[model], pad=8)
        top_ax.text(
            0.03,
            0.95,
            CASE_STRATEGY_NOTES[model],
            transform=top_ax.transAxes,
            ha="left",
            va="top",
            fontsize=9.5,
            color=MODEL_COLORS[model],
            fontweight="semibold",
            bbox={
                "boxstyle": "round,pad=0.28,rounding_size=0.15",
                "fc": lighten_color(MODEL_COLORS[model], 0.86),
                "ec": "none",
                "alpha": 0.96,
            },
            zorder=7,
        )
        top_ax.text(
            0.03,
            0.78,
            _case_summary_text(model, metrics),
            transform=top_ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.6,
            color="#2B2B2B",
            linespacing=1.35,
            bbox={
                "boxstyle": "round,pad=0.26,rounding_size=0.14",
                "fc": "#FFFDF8",
                "ec": "#DCCFBE",
                "lw": 0.7,
                "alpha": 0.97,
            },
            zorder=7,
        )
        if col_idx == 0:
            top_ax.set_ylabel("Cooling-side power (MW)")

        bottom_ax.axhline(0.0, color="#CDBFAE", linewidth=0.9, zorder=1)
        bottom_ax.plot(
            x,
            subset["gt_power_mw"],
            color=CASE_SIGNAL_COLORS["gt"],
            linewidth=1.9,
            label="GT power",
        )
        bottom_ax.plot(
            x,
            subset["grid_net_mw"],
            color=CASE_SIGNAL_COLORS["grid"],
            linewidth=1.8,
            label="Net grid",
        )
        storage_ax.plot(
            x,
            subset["tes_energy_mwh"],
            color=CASE_SIGNAL_COLORS["tes"],
            linewidth=1.8,
            linestyle=(0, (2, 2)),
            label="TES energy",
        )
        _style_axes(bottom_ax, grid_axis="y")
        bottom_ax.spines["right"].set_visible(False)
        storage_ax.spines["top"].set_visible(False)
        storage_ax.spines["left"].set_visible(False)
        storage_ax.spines["right"].set_linewidth(0.9)
        storage_ax.spines["right"].set_color("#CDBFAE")
        storage_ax.tick_params(colors="#7C6340", length=3.0, width=0.8)
        if col_idx == 0:
            bottom_ax.set_ylabel("Electric / thermal dispatch (MW)")
            storage_ax.set_ylabel("TES energy (MWh)", color="#7C6340")
        else:
            storage_ax.set_yticklabels([])

        tick_positions = np.linspace(0, len(subset) - 1, 5, dtype=int)
        tick_labels = [subset["timestamp"].iloc[idx].strftime("%m-%d\n%H:%M") for idx in tick_positions]
        bottom_ax.set_xticks(tick_positions, tick_labels)

    legend_handles = [
        Line2D([0], [0], color=CASE_SIGNAL_COLORS["demand"], lw=1.8, linestyle=(0, (4, 2)), label="Cooling demand"),
        Line2D([0], [0], color=CASE_SIGNAL_COLORS["abs"], lw=2.0, label="ABS cooling"),
        Line2D([0], [0], color=CASE_SIGNAL_COLORS["ech"], lw=2.0, label="ECH cooling"),
        Line2D([0], [0], color=CASE_SIGNAL_COLORS["gt"], lw=1.9, label="GT power"),
        Line2D([0], [0], color=CASE_SIGNAL_COLORS["grid"], lw=1.8, label="Net grid import"),
        Line2D([0], [0], color=CASE_SIGNAL_COLORS["tes"], lw=1.8, linestyle=(0, (2, 2)), label="TES energy"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=6,
        bbox_to_anchor=(0.5, 0.98),
        columnspacing=1.2,
        handlelength=2.2,
    )
    fig.suptitle("Representative high-cooling-load dispatch window", fontsize=14.5, fontweight="semibold", color="#1D232A", y=0.95)
    fig.text(
        0.07,
        0.06,
        "The 24 h window is selected by the highest rolling-average cooling demand on the common 2025 horizon.",
        fontsize=9.0,
        color="#5F5A53",
        ha="left",
    )
    fig.text(
        0.07,
        0.025,
        "The annotations summarize the within-window mechanism that aligns with the yearly ranking: rbDQN is steadier, DDPG is cheaper, and PPO leans more on ECH.",
        fontsize=9.0,
        color="#5F5A53",
        ha="left",
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate publication-facing multi-seed figures from results-only archives.")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT, help="Root directory for results outputs.")
    parser.add_argument("--archive-tag", default=DEFAULT_ARCHIVE_TAG, help="Archive tag under results/archive/ to read.")
    parser.add_argument("--full-table", type=Path, default=DEFAULT_FULL_TABLE, help="Full per-run DRL yearly-eval table.")
    parser.add_argument("--aggregate-table", type=Path, default=DEFAULT_AGG_TABLE, help="Aggregate DRL yearly-eval table.")
    args = parser.parse_args()

    _set_theme()
    figure_dirs = _figure_dirs(args.results_root)
    bundle = _load_bundle(
        args.results_root,
        archive_tag=args.archive_tag,
        full_table=args.full_table,
        agg_table=args.aggregate_table,
    )
    _export_plot_tables(bundle, args.results_root)

    figures = {
        "multi_seed_frontier": plot_multi_seed_frontier(bundle),
        "multi_seed_cost_strip": plot_multi_seed_cost_strip(bundle),
        "multi_seed_training_cost": plot_training_cost(bundle),
        "multi_seed_training_cooling": plot_training_cooling(bundle),
        "multi_seed_cooling_strip": plot_multi_seed_cooling_strip(bundle),
        "multi_seed_case_window": plot_case_window(bundle, args.results_root),
    }
    for stem, fig in figures.items():
        _save_figure(fig, stem, figure_dirs)
        plt.close(fig)


if __name__ == "__main__":
    main()
