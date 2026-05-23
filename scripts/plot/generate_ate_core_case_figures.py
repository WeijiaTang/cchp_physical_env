"""Generate compact ATE revision figures from existing annual-evaluation artifacts.

The script intentionally uses only archived outputs; it does not launch new
simulation or training runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "results" / "archive" / "paper_ready_2026-03-31"
TABLE_DIR = ROOT / "results" / "tables" / "paper"
OUT_DIR = ROOT / "cchp-paper" / "figures" / "generated"

BASELINE_AGG = TABLE_DIR / "drl_all_available_multi_seed_yearly_eval_aggregate_2026-03-31.csv"
DPAR_RUNS = TABLE_DIR / "dpar_multi_seed_anchor_pool_2026-04-21.csv"
ABLATION = TABLE_DIR / "ate_revision_dpar_ablation_seed0.csv"
SURROGATE = TABLE_DIR / "ate_revision_surrogate_audit.csv"
TAIL_RISK = TABLE_DIR / "ate_revision_cooling_tail_risk.csv"
DPAR_REP_STEP_LOG = ARCHIVE / "drl" / "dpar" / "seed_114514" / "eval" / "step_log_light.csv"

REPRESENTATIVE_STEP_LOGS = {
    "DPAR": DPAR_REP_STEP_LOG,
    "rbDQN": ARCHIVE / "drl" / "dqn" / "seed_42" / "eval" / "step_log_light.csv",
    "DDPG": ARCHIVE / "drl" / "ddpg" / "seed_1027" / "eval" / "step_log_light.csv",
    "SAC": ARCHIVE / "drl" / "sac" / "seed_1" / "eval" / "step_log_light.csv",
    "Rule": ARCHIVE / "baselines" / "rule_reference_h16" / "eval" / "step_log_light.csv",
}

METHOD_ORDER = [
    "Archived rule reference",
    "rbDQN",
    "DDPG+rule residual",
    "TD3+rule residual",
    "SAC+rule residual",
    "PPO+rule residual",
    "DPAR",
]

COLORS = {
    "Archived rule reference": "#8C8C8C",
    "rbDQN": "#4C78A8",
    "DDPG+rule residual": "#F58518",
    "TD3+rule residual": "#E45756",
    "SAC+rule residual": "#72B7B2",
    "PPO+rule residual": "#B279A2",
    "DPAR": "#2F9E44",
}

COMPONENT_COLORS = {
    "Grid net": "#9ECAE1",
    "GT fuel/O&M": "#F6C85F",
    "Boiler": "#F29E4C",
    "Carbon": "#8AB17D",
    "Storage": "#B8B8D1",
    "Penalties": "#D95F59",
}


def _set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "font.size": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.dpi": 600,
        }
    )


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def _friendly_model(label: str) -> str:
    return label.replace("+rule_residual", "+rule residual")


def _summary_paths() -> dict[str, list[Path]]:
    baseline_catalog = pd.read_csv(ARCHIVE / "catalogs" / "drl_eval_catalog.csv")
    dpar_catalog = pd.read_csv(ARCHIVE / "catalogs" / "dpar_eval_catalog.csv")
    paths: dict[str, list[Path]] = {m: [] for m in METHOD_ORDER}
    for _, row in baseline_catalog.iterrows():
        model = _friendly_model(str(row["model"]))
        if model in paths:
            paths[model].append(Path(str(row["archive_dir"])) / "eval" / "summary.json")
    for _, row in dpar_catalog.iterrows():
        paths["DPAR"].append(Path(str(row["archive_dir"])) / "eval" / "summary.json")
    paths["Archived rule reference"].append(
        ARCHIVE / "baselines" / "rule_reference_h16" / "eval" / "summary.json"
    )
    return paths


def _component_groups(cost: dict) -> dict[str, float]:
    grid_net = (
        float(cost.get("grid_import", 0.0))
        - float(cost.get("grid_export_revenue", 0.0))
        + float(cost.get("grid_curtail", 0.0))
    )
    gt = float(cost.get("gt_fuel", 0.0)) + float(cost.get("gt_om", 0.0))
    boiler = float(cost.get("boiler", 0.0))
    carbon = float(cost.get("carbon", 0.0))
    storage = float(cost.get("bes_degr", 0.0))
    penalties = sum(
        float(cost.get(k, 0.0))
        for k in [
            "grid_export_penalty",
            "unmet_e",
            "unmet_h",
            "unmet_c",
            "viol",
            "invalid_abs_request",
            "gt_toggle",
            "gt_delta",
            "idle_heat_backup",
            "idle_cool_backup",
        ]
    )
    return {
        "Grid net": grid_net / 1e6,
        "GT fuel/O&M": gt / 1e6,
        "Boiler": boiler / 1e6,
        "Carbon": carbon / 1e6,
        "Storage": storage / 1e6,
        "Penalties": penalties / 1e6,
    }


def _annual_stats() -> pd.DataFrame:
    agg = pd.read_csv(BASELINE_AGG)
    rows = []
    for _, row in agg.iterrows():
        model = _friendly_model(str(row["model"]))
        rows.append(
            {
                "method": model,
                "n": int(row["n_seeds"]),
                "cost_mean": float(row["total_cost_mean"]),
                "cost_std": float(row["total_cost_std"]),
                "cool_mean": float(row["rel_cool_mean"]),
                "cool_std": float(row["rel_cool_std"]),
                "gt_mean": float(row["starts_gt_mean"]),
                "gt_std": float(row["starts_gt_std"]),
            }
        )
    dpar = pd.read_csv(DPAR_RUNS)
    rows.append(
        {
            "method": "DPAR",
            "n": len(dpar),
            "cost_mean": float(dpar["total_cost_m"].mean()),
            "cost_std": float(dpar["total_cost_m"].std(ddof=1)),
            "cool_mean": float(dpar["rel_cool"].mean()),
            "cool_std": float(dpar["rel_cool"].std(ddof=1)),
            "gt_mean": float(dpar["starts_gt"].mean()),
            "gt_std": float(dpar["starts_gt"].std(ddof=1)),
        }
    )
    rule = _load_json(ARCHIVE / "baselines" / "rule_reference_h16" / "eval" / "summary.json")
    rows.append(
        {
            "method": "Archived rule reference",
            "n": 1,
            "cost_mean": float(rule["total_cost"]) / 1e6,
            "cost_std": 0.0,
            "cool_mean": float(rule["reliability"]["cooling"]),
            "cool_std": 0.0,
            "gt_mean": float(rule["starts"]["gt"]),
            "gt_std": 0.0,
        }
    )
    out = pd.DataFrame(rows)
    out["method"] = pd.Categorical(out["method"], METHOD_ORDER, ordered=True)
    return out.sort_values("method").reset_index(drop=True)


def plot_frontier() -> None:
    df = _annual_stats()
    fig, ax = plt.subplots(figsize=(7.1, 4.65))
    for _, row in df.iterrows():
        method = str(row["method"])
        size = 34 + 0.105 * float(row["gt_mean"])
        ax.errorbar(
            row["cool_mean"],
            row["cost_mean"],
            xerr=row["cool_std"] if row["n"] > 1 else None,
            yerr=row["cost_std"] if row["n"] > 1 else None,
            fmt="none",
            ecolor=COLORS[method],
            elinewidth=0.9,
            alpha=0.75,
            capsize=2.2,
            zorder=1,
        )
        ax.scatter(
            row["cool_mean"],
            row["cost_mean"],
            s=size,
            color=COLORS[method],
            edgecolor="white",
            linewidth=0.8,
            label=method,
            zorder=2,
        )
    ax.axvline(0.99, color="#333333", lw=0.9, ls="--")
    ax.text(0.99015, ax.get_ylim()[0] + 0.05, "0.99 cooling gate", rotation=90, va="bottom", fontsize=7)
    ax.set_xlabel("Annual cooling reliability")
    ax.set_ylabel("Annual total cost (million CNY-eq.)")
    ax.set_xlim(0.9885, 1.0007)
    ax.grid(True, color="#E6E6E6", lw=0.7)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7, title="Bubble: GT starts")
    _save(fig, "multi_seed_frontier")
    df.to_csv(OUT_DIR / "multi_seed_frontier_core_data.csv", index=False)


def plot_cost_decomposition() -> None:
    rows = []
    for method, paths in _summary_paths().items():
        comp_rows = []
        for path in paths:
            if path.exists():
                comp_rows.append(_component_groups(_load_json(path)["cost_breakdown"]))
        if not comp_rows:
            continue
        mean_comp = pd.DataFrame(comp_rows).mean().to_dict()
        rows.append({"method": method, **mean_comp})
    df = pd.DataFrame(rows)
    df["method"] = pd.Categorical(df["method"], METHOD_ORDER, ordered=True)
    df = df.sort_values("method").reset_index(drop=True)

    # Merge the visually small storage term into a compact "Other penalty" band.
    df["Other penalty"] = df["Storage"] + df["Penalties"]
    plot_components = ["Grid net", "GT fuel/O&M", "Boiler", "Carbon", "Other penalty"]
    plot_colors = {
        "Grid net": COMPONENT_COLORS["Grid net"],
        "GT fuel/O&M": COMPONENT_COLORS["GT fuel/O&M"],
        "Boiler": COMPONENT_COLORS["Boiler"],
        "Carbon": COMPONENT_COLORS["Carbon"],
        "Other penalty": "#C75D5D",
    }

    fig, ax = plt.subplots(figsize=(7.25, 4.15))
    y = np.arange(len(df))
    left = np.zeros(len(df))
    for comp in plot_components:
        vals = df[comp].to_numpy(float)
        ax.barh(
            y,
            vals,
            left=left,
            color=plot_colors[comp],
            edgecolor="white",
            linewidth=0.55,
            height=0.72,
            label=comp,
        )
        left += vals

    dpar_idx = df.index[df["method"].astype(str) == "DPAR"][0]
    ax.axhspan(dpar_idx - 0.45, dpar_idx + 0.45, color="#EAF6EF", zorder=-1)
    ax.barh(
        dpar_idx,
        left[dpar_idx],
        left=0,
        facecolor="none",
        edgecolor="#1B7F3A",
        linewidth=1.4,
        height=0.78,
    )

    labels = [m.replace("Archived rule reference", "Rule ref.").replace("+rule residual", "+res.") for m in df["method"]]
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Annual cost (million CNY-eq.)")
    ax.set_title("Annual cost structure", loc="left", fontsize=9.2, pad=5, fontweight="semibold")
    ax.grid(axis="x", color="#E8E8E8", lw=0.7)
    ax.legend(ncols=5, loc="lower center", bbox_to_anchor=(0.5, 1.02), fontsize=6.8, columnspacing=0.9, handlelength=1.1)
    for i, total in enumerate(left):
        weight = "semibold" if i == dpar_idx else "normal"
        color = "#1B7F3A" if i == dpar_idx else "#333333"
        ax.text(total + 0.18, i, f"{total:.1f}", va="center", ha="left", fontsize=7.2, fontweight=weight, color=color)

    try:
        rb_idx = df.index[df["method"].astype(str) == "rbDQN"][0]
        delta = left[dpar_idx] - left[rb_idx]
        ax.annotate(
            f"{delta:.2f} M vs rbDQN",
            xy=(left[dpar_idx], dpar_idx),
            xytext=(left[dpar_idx] + 2.4, dpar_idx - 0.55),
            arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "#1B7F3A"},
            color="#1B7F3A",
            fontsize=7.2,
            ha="left",
        )
    except Exception:
        pass
    ax.set_xlim(0, max(left) + 3.0)
    _save(fig, "annual_cost_decomposition")
    df.to_csv(OUT_DIR / "annual_cost_decomposition_data.csv", index=False)


def plot_ablation_heatmap() -> None:
    df = pd.read_csv(ABLATION)
    order = [
        "DQN anchor only",
        "DPAR: boiler refinement",
        "DPAR: ABS refinement",
        "DPAR: GT refinement",
        "PAFC-TD3 only",
        "All-channel replacement",
    ]
    df["Variant"] = pd.Categorical(df["Variant"], order, ordered=True)
    df = df.sort_values("Variant").reset_index(drop=True)
    base = df[df["Variant"].astype(str) == "DQN anchor only"].iloc[0]

    short_names = {
        "DQN anchor only": "Anchor",
        "DPAR: boiler refinement": "Boiler",
        "DPAR: ABS refinement": "ABS",
        "DPAR: GT refinement": "GT",
        "PAFC-TD3 only": "PAFC",
        "All-channel replacement": "All",
    }

    fig = plt.figure(figsize=(7.25, 3.25))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.18, 1.0], wspace=0.33)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    display_order = [
        "DPAR: boiler refinement",
        "DPAR: ABS refinement",
        "DQN anchor only",
        "DPAR: GT refinement",
        "PAFC-TD3 only",
        "All-channel replacement",
    ]
    p = df.set_index(df["Variant"].astype(str)).loc[display_order].reset_index(drop=True)
    y0 = np.arange(len(p))
    bar_colors = [
        COLORS["DPAR"] if str(v) == "DPAR: boiler refinement" else "#D95F59" if "All-channel" in str(v) else "#AEB6BE"
        for v in p["Variant"]
    ]
    ax0.barh(y0, p["Cost"], color=bar_colors, height=0.58, edgecolor="white", linewidth=0.7)
    ax0.axvline(float(base["Cost"]), color="#4C4C4C", lw=0.9, ls="--")
    ax0.set_yticks(y0, [short_names[str(v)] for v in p["Variant"]])
    ax0.invert_yaxis()
    ax0.set_xlim(0, 26.6)
    ax0.set_xlabel("Annual cost (million CNY-eq.)")
    ax0.set_title("(a) Cost--reliability screen", loc="left", fontsize=8.8, fontweight="semibold")
    ax0.grid(axis="x", color="#E8E8E8", lw=0.7)
    for yi, (variant, cost, rcool) in enumerate(zip(p["Variant"], p["Cost"], p["Rcool"])):
        is_retained = str(variant) == "DPAR: boiler refinement"
        ax0.text(
            float(cost) + 0.22,
            yi,
            f"{float(cost):.1f}",
            va="center",
            fontsize=6.9,
            fontweight="semibold" if is_retained else "normal",
            color="#1B7F3A" if is_retained else "#333333",
        )
        ax0.text(
            26.15,
            yi,
            f"{float(rcool):.5f}",
            va="center",
            ha="right",
            fontsize=6.3,
            color="#1B7F3A" if is_retained else "#5B6167",
        )
    ax0.text(26.15, -0.72, "$R_c$", ha="right", va="center", fontsize=6.7, color="#5B6167")
    ax0.add_patch(
        mpl.patches.Rectangle((0.05, -0.38), 16.15, 0.76, fill=False, ec="#1B7F3A", lw=1.1, zorder=3)
    )

    retained = df[df["Variant"].astype(str) == "DPAR: boiler refinement"].iloc[0]
    deltas = {
        "Annual cost": ("Cost", True),
        "GT starts": ("GT", False),
        "Boiler cost": ("BoilerCost", True),
        "Carbon cost": ("CarbonCost", True),
    }
    labels = list(deltas.keys())
    vals = []
    for label, (col, _) in deltas.items():
        denom = float(base[col]) if abs(float(base[col])) > 1e-9 else 1.0
        vals.append(100.0 * (float(retained[col]) - float(base[col])) / denom)
    y = np.arange(len(labels))
    bar_colors = ["#1B7F3A" if v < 0 else "#D9843B" for v in vals]
    bars = ax1.barh(y, vals, color=bar_colors, height=0.52, edgecolor="white", linewidth=0.8)
    ax1.axvline(0, color="#3A3A3A", lw=0.9)
    ax1.set_yticks(y, labels)
    ax1.tick_params(axis="y", pad=7)
    ax1.invert_yaxis()
    ax1.set_xlabel("Change vs DQN anchor (%)")
    ax1.set_title("(b) Retained-case mechanism", loc="left", fontsize=8.8, fontweight="semibold")
    ax1.grid(axis="x", color="#E8E8E8", lw=0.7)
    ax1.set_xlim(-18, 48)
    for bar, val in zip(bars, vals):
        if val < 0:
            ax1.text(val / 2, bar.get_y() + bar.get_height() / 2, f"{val:+.1f}", va="center", ha="center", fontsize=7.0, color="white")
        else:
            ax1.text(val + 1.2, bar.get_y() + bar.get_height() / 2, f"{val:+.1f}", va="center", ha="left", fontsize=7.0)
    _save(fig, "dpar_ablation_heatmap")


def plot_comprehensive_mechanism_audit() -> None:
    """Create a compact radar summary of DPAR mechanism and surrogate risk."""
    ab = pd.read_csv(ABLATION)
    su = pd.read_csv(SURROGATE)
    channel_map = {
        "u_gt": "GT",
        "u_bes": "BES",
        "u_boiler": "Boiler",
        "u_abs": "ABS",
        "u_ech": "ECH",
        "u_tes": "TES",
    }
    su["Channel"] = su["Action"].map(channel_map)
    rmse = su.set_index("Channel")["DeployRMSE"].astype(float).to_dict()
    max_rmse = max(rmse.values())

    variants = [
        ("DQN anchor", "DQN anchor only", None, "#6C757D", 1.0),
        ("Boiler refinement", "DPAR: boiler refinement", "Boiler", COLORS["DPAR"], 2.4),
        ("ABS refinement", "DPAR: ABS refinement", "ABS", "#4C78A8", 1.2),
        ("GT refinement", "DPAR: GT refinement", "GT", "#F58518", 1.2),
        ("All-channel", "All-channel replacement", "All", "#D95F59", 1.2),
    ]
    selected = ab.set_index("Variant").loc[[v[1] for v in variants]].reset_index()
    cost_min = float(selected["Cost"].min())
    cost_max = float(selected["Cost"].max())
    unmet_max = max(float(selected["Unmet"].max()), 1e-9)

    def _surrogate_conf(channel: str | None) -> float:
        if channel is None:
            return 1.0
        if channel == "All":
            return 0.0
        return float(np.clip(1.0 - rmse[channel] / max_rmse, 0.0, 1.0))

    def _sparse_score(channel: str | None) -> float:
        return 0.0 if channel == "All" else 1.0

    def _anchor_preservation(channel: str | None) -> float:
        if channel is None:
            return 1.0
        if channel == "All":
            return 0.0
        return 5.0 / 6.0

    axes_labels = [
        "Low\nannual cost",
        "Cooling\nadequacy",
        "Low unmet\ncooling",
        "Surrogate\nconfidence",
        "Sparse\nmask",
        "Anchor\npreservation",
    ]

    rows = []
    for label, variant_name, channel, color, lw in variants:
        row = selected[selected["Variant"].astype(str) == variant_name].iloc[0]
        scores = {
            "Low\nannual cost": (cost_max - float(row["Cost"])) / max(cost_max - cost_min, 1e-9),
            "Cooling\nadequacy": np.clip((float(row["Rcool"]) - 0.99) / 0.01, 0.0, 1.0),
            "Low unmet\ncooling": 1.0 - float(row["Unmet"]) / unmet_max,
            "Surrogate\nconfidence": _surrogate_conf(channel),
            "Sparse\nmask": _sparse_score(channel),
            "Anchor\npreservation": _anchor_preservation(channel),
        }
        rows.append({"label": label, "variant": variant_name, "channel": channel or "None", **scores})

    radar = pd.DataFrame(rows)
    angles = np.linspace(0, 2 * np.pi, len(axes_labels), endpoint=False)
    angles_closed = np.r_[angles, angles[0]]

    fig = plt.figure(figsize=(5.25, 3.35))
    ax = fig.add_axes([0.07, 0.15, 0.58, 0.72], polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1.0)
    ax.set_facecolor("#FBFCFD")
    ax.spines["polar"].set_color("#BFC7D0")
    ax.spines["polar"].set_linewidth(0.8)
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=6.2, color="#7A828A")
    ax.yaxis.grid(True, color="#DCE2E8", lw=0.65, ls="--")
    ax.xaxis.grid(True, color="#E2E7EC", lw=0.65)
    ax.set_xticks(angles)
    ax.set_xticklabels(axes_labels, fontsize=7.4, fontweight="semibold")

    for label, variant_name, channel, color, lw in variants:
        vals = radar.loc[radar["label"] == label, axes_labels].iloc[0].astype(float).to_numpy()
        vals_closed = np.r_[vals, vals[0]]
        alpha = 0.18 if label == "Boiler refinement" else 0.045
        z = 6 if label == "Boiler refinement" else 3
        ax.plot(angles_closed, vals_closed, color=color, lw=lw, label=label, zorder=z)
        ax.fill(angles_closed, vals_closed, color=color, alpha=alpha, zorder=z - 1)
        if label == "Boiler refinement":
            ax.scatter(angles, vals, s=20, color=color, edgecolor="white", linewidth=0.55, zorder=8)

    ax.text(
        0.5,
        0.5,
        "Retained\nDPAR mask",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8.2,
        fontweight="semibold",
        color="#1B7F3A",
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#DCE8DE", "lw": 0.7},
    )
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.08, 0.50),
        ncols=1,
        fontsize=6.5,
        columnspacing=0.9,
        handlelength=1.7,
    )
    fig.text(
        0.70,
        0.18,
        "Normalized diagnostic axes\n(larger is more favorable)",
        ha="left",
        va="center",
        fontsize=6.4,
        color="#5B6167",
    )
    _save(fig, "dpar_comprehensive_audit")
    radar.to_csv(OUT_DIR / "dpar_comprehensive_audit_radar_data.csv", index=False)

def plot_surrogate_mask() -> None:
    df = pd.read_csv(SURROGATE)
    labels = ["GT", "BES", "Boiler", "ABS", "ECH", "TES"]
    mae = df["DeployMAE"].to_numpy(float)
    rmse = df["DeployRMSE"].to_numpy(float)
    risk = 0.5 * (mae + rmse)
    order = np.argsort(risk)
    labels_o = [labels[i] for i in order]
    mae_o = mae[order]
    rmse_o = rmse[order]
    risk_o = risk[order]
    y = np.arange(len(order))

    fig = plt.figure(figsize=(7.15, 3.25))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    decision_ax = fig.add_subplot(gs[0, 1])

    for yi, lab in zip(y, labels_o):
        if lab == "Boiler":
            color = "#E7F4EA"
        elif lab == "ECH":
            color = "#FBEAEA"
        else:
            color = "#FFFFFF"
        ax.axhspan(yi - 0.43, yi + 0.43, color=color, zorder=0)

    point_colors = [
        "#1B7F3A" if lab == "Boiler" else "#D95F59" if lab == "ECH" else "#7C8793"
        for lab in labels_o
    ]
    ax.hlines(y, mae_o, rmse_o, color="#AEB4BB", lw=2.0, zorder=2)
    ax.scatter(mae_o, y, marker="o", s=42, color=point_colors, edgecolor="white", linewidth=0.8, label="MAE", zorder=3)
    ax.scatter(rmse_o, y, marker="D", s=38, color=point_colors, edgecolor="white", linewidth=0.8, label="RMSE", zorder=3)
    ax.set_yticks(y, labels_o)
    ax.set_xlabel("Normalized deployment error")
    ax.set_xlim(0.13, 1.03)
    ax.set_title("(a) Deployment error by channel", loc="left", fontsize=8.8, fontweight="semibold")
    ax.grid(axis="x", color="#E8E8E8", lw=0.7)
    ax.legend(loc="lower right", fontsize=6.8, handlelength=1.1)
    for yi, lab, m, r in zip(y, labels_o, mae_o, rmse_o):
        if lab in ["Boiler", "ECH"]:
            ax.text(r + 0.025, yi, f"{r:.2f}", va="center", fontsize=6.8, color="#333333")

    decision_ax.set_xlim(0, 1)
    decision_ax.set_ylim(0, 1)
    card_w, card_h = 0.27, 0.23
    x0s = [0.05, 0.365, 0.68]
    y0s = [0.58, 0.24]
    card_order = ["GT", "BES", "Boiler", "ABS", "ECH", "TES"]
    for idx, lab in enumerate(card_order):
        row, col = divmod(idx, 3)
        x0 = x0s[col]
        y0 = y0s[row]
        is_boiler = lab == "Boiler"
        is_ech = lab == "ECH"
        fc = "#E7F4EA" if is_boiler else "#F6F7F8"
        ec = "#1B7F3A" if is_boiler else "#D8DDE2"
        accent = "#1B7F3A" if is_boiler else "#D95F59" if is_ech else "#A8B0B8"
        status = "REFINE" if is_boiler else "ANCHOR"
        patch = mpl.patches.FancyBboxPatch(
            (x0, y0),
            card_w,
            card_h,
            boxstyle="round,pad=0.025,rounding_size=0.08",
            facecolor=fc,
            edgecolor=ec,
            linewidth=1.1,
        )
        decision_ax.add_patch(patch)
        decision_ax.add_patch(
            mpl.patches.FancyBboxPatch(
                (x0, y0),
                0.025,
                card_h,
                boxstyle="round,pad=0.0,rounding_size=0.05",
                facecolor=accent,
                edgecolor=accent,
                linewidth=0,
            )
        )
        decision_ax.text(
            x0 + card_w / 2,
            y0 + 0.145,
            lab,
            ha="center",
            va="center",
            fontsize=8.1,
            fontweight="semibold",
            color="#263238",
        )
        decision_ax.text(
            x0 + card_w / 2,
            y0 + 0.065,
            status,
            ha="center",
            va="center",
            fontsize=6.3,
            color="#1B7F3A" if is_boiler else "#737B83",
        )
    decision_ax.annotate(
        "",
        xy=(0.815, 0.55),
        xytext=(0.815, 0.48),
        arrowprops={"arrowstyle": "-|>", "lw": 0.9, "color": "#1B7F3A"},
    )
    decision_ax.text(
        0.5,
        0.08,
        "sparse refinement; other commands remain anchor-controlled",
        ha="center",
        va="center",
        fontsize=6.7,
        color="#4B535B",
    )
    decision_ax.set_yticks([])
    decision_ax.set_xticks([])
    decision_ax.set_title("(b) Retained sparse mask", loc="left", fontsize=8.8, fontweight="semibold")
    for spine in decision_ax.spines.values():
        spine.set_visible(False)
    _save(fig, "surrogate_refinement_mask")


def plot_diagnostic_surrogate_summary() -> None:
    """Merge the ablation diagnostic and surrogate audit into one compact figure.

    The combined panel is intended for the main manuscript. It keeps only two
    messages: (a) the retained sparse channel is the low-cost diagnostic case;
    (b) the surrogate audit motivates a sparse mask rather than all-channel use.
    """
    ab = pd.read_csv(ABLATION)
    order = [
        "DPAR: boiler refinement",
        "DPAR: ABS refinement",
        "DQN anchor only",
        "DPAR: GT refinement",
        "PAFC-TD3 only",
        "All-channel replacement",
    ]
    short_names = {
        "DQN anchor only": "Anchor",
        "DPAR: boiler refinement": "Boiler",
        "DPAR: ABS refinement": "ABS",
        "DPAR: GT refinement": "GT",
        "PAFC-TD3 only": "PAFC",
        "All-channel replacement": "All",
    }
    ab = ab.set_index("Variant").loc[order].reset_index()
    base_cost = float(ab.loc[ab["Variant"] == "DQN anchor only", "Cost"].iloc[0])
    y = np.arange(len(ab))

    su = pd.read_csv(SURROGATE)
    channel_map = {
        "u_gt": "GT",
        "u_bes": "BES",
        "u_boiler": "Boiler",
        "u_abs": "ABS",
        "u_ech": "ECH",
        "u_tes": "TES",
    }
    su["Channel"] = su["Action"].map(channel_map)
    su["Risk"] = 0.5 * (su["DeployMAE"] + su["DeployRMSE"])
    su_order = ["ECH", "BES", "Boiler", "ABS", "TES", "GT"]
    su = su.set_index("Channel").loc[su_order].reset_index()
    sy = np.arange(len(su))

    fig = plt.figure(figsize=(7.45, 3.45))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.04, 1.0], wspace=0.40)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    # Panel (a): cost ladder around the DQN anchor.
    ax0.axvspan(15.75, 16.55, color="#EAF6EF", zorder=0)
    ax0.axvline(base_cost, color="#404040", lw=0.9, ls="--", zorder=1)
    for yi, row in ab.iterrows():
        name = str(row["Variant"])
        cost = float(row["Cost"])
        color = COLORS["DPAR"] if name == "DPAR: boiler refinement" else "#D95F59" if "All-channel" in name else "#AEB6BE"
        ax0.hlines(yi, min(base_cost, cost), max(base_cost, cost), color=color, lw=3.0, alpha=0.72, zorder=2)
        ax0.scatter(cost, yi, s=58 if name == "DPAR: boiler refinement" else 42, color=color, edgecolor="white", linewidth=0.85, zorder=3)
        ax0.text(cost + 0.10, yi, f"{cost:.1f}", va="center", ha="left", fontsize=6.8,
                 color="#1B7F3A" if name == "DPAR: boiler refinement" else "#333333",
                 fontweight="semibold" if name == "DPAR: boiler refinement" else "normal")
    ax0.set_yticks(y, [short_names[str(v)] for v in ab["Variant"]])
    ax0.invert_yaxis()
    ax0.set_xlim(15.55, 21.45)
    ax0.set_xlabel("Annual cost (million CNY-eq.)")
    ax0.set_title("(a) Single-seed channel ablation", loc="left", fontsize=8.8, fontweight="semibold")
    ax0.grid(axis="x", color="#E8E8E8", lw=0.7)
    ax0.text(
        0.98,
        0.96,
        r"all variants pass $R_c \geq 0.99$",
        transform=ax0.transAxes,
        ha="right",
        va="top",
        fontsize=6.5,
        color="#5B6167",
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#E0E0E0", "lw": 0.6},
    )

    # Panel (b): surrogate error with the retained mask overlaid.
    for yi, ch in zip(sy, su["Channel"]):
        if ch == "Boiler":
            fc = "#E7F4EA"
        elif ch == "ECH":
            fc = "#FBEAEA"
        else:
            fc = "#FFFFFF"
        ax1.axhspan(yi - 0.43, yi + 0.43, color=fc, zorder=0)

    point_colors = [
        "#1B7F3A" if ch == "Boiler" else "#D95F59" if ch == "ECH" else "#7C8793"
        for ch in su["Channel"]
    ]
    ax1.hlines(sy, su["DeployMAE"], su["DeployRMSE"], color="#AEB4BB", lw=2.1, zorder=2)
    ax1.scatter(su["DeployMAE"], sy, marker="o", s=36, color=point_colors, edgecolor="white", linewidth=0.75, label="MAE", zorder=3)
    ax1.scatter(su["DeployRMSE"], sy, marker="D", s=34, color=point_colors, edgecolor="white", linewidth=0.75, label="RMSE", zorder=3)
    for yi, ch, rmse in zip(sy, su["Channel"], su["DeployRMSE"]):
        status = "REFINE" if ch == "Boiler" else "ANCHOR"
        color = "#1B7F3A" if ch == "Boiler" else "#D95F59" if ch == "ECH" else "#7C8793"
        ax1.text(1.205, yi, status, va="center", ha="left", fontsize=6.2, color=color,
                 fontweight="semibold" if ch == "Boiler" else "normal")
        if ch in ["Boiler", "ECH"]:
            ax1.text(float(rmse) + 0.018, yi, f"{float(rmse):.2f}", va="center", fontsize=6.6, color="#333333")
    ax1.set_yticks(sy, list(su["Channel"]))
    ax1.invert_yaxis()
    ax1.set_xlim(0.13, 1.34)
    ax1.set_xlabel("Normalized deployment error")
    ax1.set_title("(b) Surrogate risk and retained mask", loc="left", fontsize=8.8, fontweight="semibold")
    ax1.grid(axis="x", color="#E8E8E8", lw=0.7)
    ax1.text(
        0.98,
        0.04,
        "MAE: circle   RMSE: diamond",
        transform=ax1.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.4,
        color="#5B6167",
    )
    for spine in ["right", "top"]:
        ax0.spines[spine].set_visible(False)
        ax1.spines[spine].set_visible(False)

    _save(fig, "dpar_diagnostic_surrogate_summary")


def plot_executed_seasonal_flows() -> None:
    """Plot monthly executed service allocation and thermal reliance diagnostics."""
    methods_to_plot = ["DPAR", "rbDQN", "DDPG", "Rule"]
    method_labels = {
        "DPAR": "DPAR",
        "rbDQN": "rbDQN",
        "DDPG": "DDPG",
        "Rule": "Rule ref.",
    }
    cols = [
        "timestamp",
        "energy_demand_h_mwh",
        "energy_demand_c_mwh",
        "energy_unmet_c_mwh",
        "q_hrsg_rec_mw",
        "q_boiler_mw",
        "q_abs_cool_mw",
        "q_ech_cool_mw",
        "q_tes_discharge_mw",
    ]
    dt = 0.25
    records = []
    annual_records = []

    for method in methods_to_plot:
        df = pd.read_csv(REPRESENTATIVE_STEP_LOGS[method], usecols=cols)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["month"] = df["timestamp"].dt.month

        # Demand-capped service allocation for visualization.
        demand_mw = df["energy_demand_c_mwh"].to_numpy(float) / dt
        unmet_mw = df["energy_unmet_c_mwh"].to_numpy(float) / dt
        served_mw = np.maximum(demand_mw - unmet_mw, 0.0)
        abs_served = np.minimum(df["q_abs_cool_mw"].to_numpy(float), served_mw)
        ech_served = np.minimum(df["q_ech_cool_mw"].to_numpy(float), np.maximum(served_mw - abs_served, 0.0))   
        residual = np.maximum(served_mw - abs_served - ech_served, 0.0)
        ech_served = ech_served + residual
        cooling_excess = np.maximum(
            df["q_abs_cool_mw"].to_numpy(float) + df["q_ech_cool_mw"].to_numpy(float) - served_mw,
            0.0,
        )
        df = df.assign(
            abs_served_mwh=abs_served * dt,
            ech_served_mwh=ech_served * dt,
            cooling_excess_mwh=cooling_excess * dt,
            hrsg_mwh=df["q_hrsg_rec_mw"] * dt,
            boiler_mwh=df["q_boiler_mw"] * dt,
            tes_dis_mwh=df["q_tes_discharge_mw"] * dt,
        )
        monthly = (
            df.groupby("month")
            .agg(
                cooling_demand_mwh=("energy_demand_c_mwh", "sum"),
                heat_demand_mwh=("energy_demand_h_mwh", "sum"),
                unmet_c_mwh=("energy_unmet_c_mwh", "sum"),
                abs_served_mwh=("abs_served_mwh", "sum"),
                ech_served_mwh=("ech_served_mwh", "sum"),
                cooling_excess_mwh=("cooling_excess_mwh", "sum"),
                hrsg_mwh=("hrsg_mwh", "sum"),
                boiler_mwh=("boiler_mwh", "sum"),
                tes_dis_mwh=("tes_dis_mwh", "sum"),
            )
            .reindex(range(1, 13), fill_value=0.0)
            .reset_index()
        )
        monthly["method"] = method
        useful = monthly["abs_served_mwh"] + monthly["ech_served_mwh"] + monthly["unmet_c_mwh"]
        thermal_total = monthly["hrsg_mwh"] + monthly["boiler_mwh"] + monthly["tes_dis_mwh"]
        monthly["ech_share_pct"] = np.where(useful > 1e-9, 100 * monthly["ech_served_mwh"] / useful, 0.0)       
        monthly["boiler_share_pct"] = np.where(thermal_total > 1e-9, 100 * monthly["boiler_mwh"] / thermal_total, 0.0)
        records.append(monthly)
        annual_records.append(
            {
                "method": method,
                "abs_served_mwh": monthly["abs_served_mwh"].sum(),
                "ech_served_mwh": monthly["ech_served_mwh"].sum(),
                "unmet_c_mwh": monthly["unmet_c_mwh"].sum(),
                "cooling_excess_mwh": monthly["cooling_excess_mwh"].sum(),
                "hrsg_mwh": monthly["hrsg_mwh"].sum(),
                "boiler_mwh": monthly["boiler_mwh"].sum(),
                "tes_dis_mwh": monthly["tes_dis_mwh"].sum(),
            }
        )

    out = pd.concat(records, ignore_index=True)
    annual = pd.DataFrame(annual_records)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_DIR / "executed_monthly_dispatch_data.csv", index=False)
    annual.to_csv(OUT_DIR / "executed_annual_dispatch_mix_data.csv", index=False)

    month_labels = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
    method_names = [method_labels[m] for m in methods_to_plot]
    ech_matrix = np.vstack([out[out["method"] == m].sort_values("month")["ech_share_pct"].to_numpy(float) for m in methods_to_plot])
    boiler_matrix = np.vstack([out[out["method"] == m].sort_values("month")["boiler_share_pct"].to_numpy(float) for m in methods_to_plot])

    cool_colors = {"ABS": "#2A9D8F", "ECH": "#3A73B8", "Unmet": "#D95F59", "Excess": "#B7B7B7"}
    heat_colors = {"HRSG": "#D8A21B", "Boiler": "#E76F51", "TES discharge": "#7B61A8"}

    fig = plt.figure(figsize=(7.45, 5.15))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], hspace=0.48, wspace=0.34)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    im0 = ax0.imshow(ech_matrix, aspect="auto", cmap="Blues", vmin=0, vmax=100)
    im1 = ax1.imshow(boiler_matrix, aspect="auto", cmap="Oranges", vmin=0, vmax=100)
    for ax, title in [(ax0, "(a) ECH share of useful cooling (%)"), (ax1, "(b) Boiler share of thermal-source use (%)")]:
        ax.set_xticks(np.arange(12), month_labels)
        ax.set_yticks(np.arange(len(methods_to_plot)), method_names)
        ax.tick_params(axis="both", labelsize=6.7, length=0)
        ax.set_title(title, loc="left", fontsize=8.4, fontweight="semibold")
        ax.set_xlabel("Month", fontsize=7.0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks(np.arange(-0.5, 12, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(methods_to_plot), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)
    for ax, im in [(ax0, im0), (ax1, im1)]:
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.ax.tick_params(labelsize=6.2, length=2)
        cb.outline.set_linewidth(0.4)

    y = np.arange(len(methods_to_plot))
    annual = annual.set_index("method").loc[methods_to_plot].reset_index()
    left = np.zeros(len(methods_to_plot))
    for col, label, color in [
        ("abs_served_mwh", "ABS served", cool_colors["ABS"]),
        ("ech_served_mwh", "ECH served", cool_colors["ECH"]),
        ("unmet_c_mwh", "Unmet", cool_colors["Unmet"]),
    ]:
        vals = annual[col].to_numpy(float)
        ax2.barh(y, vals, left=left, color=color, edgecolor="white", linewidth=0.45, height=0.62, label=label)  
        left += vals
    # Show excess device output as a thin grey marker rather than useful service.
    excess = annual["cooling_excess_mwh"].to_numpy(float)
    ax2.scatter(left + excess, y, marker="|", s=110, color=cool_colors["Excess"], linewidth=1.4, label="Logged excess output")
    ax2.set_title("(c) Annual useful cooling allocation", loc="left", fontsize=8.4, fontweight="semibold")      
    ax2.set_xlabel("MWh", fontsize=7.0)

    left = np.zeros(len(methods_to_plot))
    for col, label, color in [
        ("hrsg_mwh", "HRSG", heat_colors["HRSG"]),
        ("boiler_mwh", "Boiler", heat_colors["Boiler"]),
        ("tes_dis_mwh", "TES discharge", heat_colors["TES discharge"]),
    ]:
        vals = annual[col].to_numpy(float)
        ax3.barh(y, vals, left=left, color=color, edgecolor="white", linewidth=0.45, height=0.62, label=label)  
        left += vals
    ax3.set_title("(d) Annual thermal-source use", loc="left", fontsize=8.4, fontweight="semibold")
    ax3.set_xlabel("MWh", fontsize=7.0)

    for ax in [ax2, ax3]:
        ax.set_yticks(y, method_names)
        ax.invert_yaxis()
        ax.grid(axis="x", color="#E8ECEF", lw=0.65)
        ax.tick_params(axis="both", labelsize=6.7, length=2.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax2.legend(loc="lower center", bbox_to_anchor=(0.5, -0.38), ncols=2, fontsize=6.3, frameon=False)
    ax3.legend(loc="lower center", bbox_to_anchor=(0.5, -0.38), ncols=2, fontsize=6.3, frameon=False)

    _save(fig, "executed_seasonal_flows")
def plot_tail_risk_audit() -> None:
    df = pd.read_csv(TAIL_RISK)
    # Clean up names for mapping
    name_map = {
        "DPAR 11-run mean": "DPAR",
        "DPAR run 1025": "DPAR",
        "rbDQN seed 42": "rbDQN",
        "DDPG seed 1027": "DDPG+rule residual",
    }
    df["color_key"] = df["Policy"].map(name_map)
    df["label"] = df["Policy"].str.replace(" seed 42", "").str.replace(" run 1025", "").str.replace(" 11-run mean", " (avg)")

    # Data for 6-axis Radar (Hexagonal Fingerprint)
    # Metrics: [Duration, Energy, P95, P99, Peak, TailFactor]
    # TailFactor = Peak / P95 (Normalized)
    df["TailFactor"] = df["MaxMW"] / df["P95CondMW"]
    radar_metrics = ["Hours", "UnmetMWh", "P95CondMW", "P99CondMW", "MaxMW", "TailFactor"]
    radar_labels = ["Duration", "Energy", "P95-Tail", "P99-Tail", "Peak", "Severity"]

    radar_data = df[radar_metrics].copy()
    for col in radar_metrics:
        radar_data[col] = radar_data[col] / radar_data[col].max()

    fig = plt.figure(figsize=(9.0, 4.5))
    plt.subplots_adjust(wspace=0.45) 

    # Panel (a): Hexagonal Reliability Fingerprint
    ax0 = fig.add_subplot(121, polar=True)
    ax0.set_facecolor("#FDFDFD")
    
    angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
    angles += angles[:1]
    
    for i, row in radar_data.iterrows():
        values = row.tolist()
        values += values[:1]
        color = COLORS.get(df.iloc[i]["color_key"], "#8C8C8C")
        ax0.plot(angles, values, color=color, linewidth=1.8, alpha=0.9, label=df.iloc[i]["label"], zorder=5)
        ax0.fill(angles, values, color=color, alpha=0.06, zorder=4)

    ax0.set_xticks(angles[:-1])
    ax0.set_xticklabels(radar_labels, fontsize=7.5, fontweight="semibold")
    ax0.yaxis.grid(True, color="#E9ECEF", linestyle="--", linewidth=0.5)
    ax0.set_yticklabels([]) 
    ax0.spines["polar"].set_color("#CED4DA")
    ax0.set_title("(a) Hexagonal Reliability Fingerprint", loc="left", fontsize=9.5, fontweight="bold", pad=35)
    ax0.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), fontsize=6.5, ncol=2, frameon=True, edgecolor="#DEE2E6")

    # Panel (b): Polar Sector Risk Landscape (Sonar Aesthetic)
    # Radius = Intensity (MW), Angle = Duration (Normalized to 0-90 deg)
    ax1 = fig.add_subplot(122, polar=True)
    ax1.set_theta_zero_location("N")
    ax1.set_theta_direction(-1)
    ax1.set_thetamin(0)
    ax1.set_thetamax(90)
    
    # Create the risk field in polar coordinates
    r_max = 2.5
    h_max = 500
    r = np.linspace(0, r_max, 100)
    theta = np.linspace(0, np.pi/2, 100)
    R, THETA = np.meshgrid(r, theta)
    # Risk Potential: sqrt(norm_intensity * norm_duration)
    Z = np.sqrt((R / 2.0) * (THETA / (np.pi/2)))
    
    ax1.contourf(THETA, R, Z, levels=25, cmap="RdYlGn_r", alpha=0.22, antialiased=True)
    iso_levels = [0.2, 0.45, 0.75, 1.1]
    contours = ax1.contour(THETA, R, Z, levels=iso_levels, colors="#6C757D", linewidths=0.4, alpha=0.25, linestyles="--")
    ax1.clabel(contours, inline=True, fontsize=5, fmt="%.1f")

    # Map policies onto the Sonar Sector
    for i, row in df.iterrows():
        color = COLORS.get(row["color_key"], "#8C8C8C")
        # Angle = Normalized Duration (0-500 -> 0-90 deg)
        theta_val = (row["Hours"] / h_max) * (np.pi/2)
        r_val = row["MaxMW"]
        
        # Glow markers
        ax1.scatter(theta_val, r_val, s=row["UnmetMWh"]*25, color=color, alpha=0.08, zorder=2)
        ax1.scatter(theta_val, r_val, s=row["UnmetMWh"]*15, color=color, alpha=0.25, zorder=2)
        ax1.scatter(theta_val, r_val, s=row["UnmetMWh"]*8, color=color, edgecolor="white", linewidth=1.0, alpha=1.0, zorder=3)
        
        # Staggered polar labels
        offset_r = 0.25 if "DPAR" in row["label"] else -0.25
        offset_theta = 0.05 if "avg" in row["label"] else -0.05
        ax1.text(theta_val + offset_theta, r_val + offset_r, row["label"], 
                 fontsize=6.8, fontweight="bold", color=color, 
                 ha="center", va="center", zorder=6,
                 bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=0.5))

    # Decorate Sonar
    ax1.set_rorigin(-0.2)
    ax1.set_rticks([0.5, 1.0, 1.5, 2.0])
    ax1.set_yticklabels(["0.5", "1.0", "1.5", "2.0 MW"], fontsize=6, color="#ADB5BD")
    ax1.set_xticks(np.linspace(0, np.pi/2, 5))
    ax1.set_xticklabels(["0", "125", "250", "375", "500 h"], fontsize=6, color="#ADB5BD")
    ax1.grid(True, color="#CED4DA", linestyle=":", alpha=0.5)
    
    # Sector Labels
    ax1.text(np.pi/4, 2.8, "Operational Risk Sector (Intensity x Duration)", fontsize=8, fontweight="bold", ha="center")
    ax1.text(0.1, 0.4, "ACUTE", fontsize=7, fontweight="black", color="#E67700", alpha=0.3, rotation=0)
    ax1.text(1.4, 2.2, "CHRONIC", fontsize=7, fontweight="black", color="#C92A2A", alpha=0.3, rotation=0)

    ax0.set_title("(a) Hexagonal Reliability Fingerprint", loc="left", fontsize=9.5, fontweight="bold", pad=35)
    ax1.set_title("(b) Polar Risk Landscape", loc="left", fontsize=9.5, fontweight="bold", pad=20)
    
    # Legend
    sizes = [5, 15, 25]
    for s in sizes:
        ax1.scatter([], [], s=s*8, color="#ADB5BD", label=f"{s} MWh")
    ax1.legend(title="Risk Energy", loc="lower right", bbox_to_anchor=(1.3, -0.1), 
               fontsize=6, title_fontsize=6.5, frameon=True, edgecolor="#F1F3F5")

    _save(fig, "cooling_tail_risk_audit")

    _save(fig, "cooling_tail_risk_audit")

def main() -> None:
    _set_style()
    plot_frontier()
    plot_cost_decomposition()
    plot_executed_seasonal_flows()
    plot_comprehensive_mechanism_audit()
    plot_tail_risk_audit()


if __name__ == "__main__":
    main()
