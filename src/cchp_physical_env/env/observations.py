# Ref: docs/spec/task.md
from __future__ import annotations

import math

import pandas as pd


def build_observation(
    *,
    row: pd.Series,
    bes_soc: float,
    gt_prev_on: bool,
    gt_state: float,
    p_gt_prev_mw: float,
    gt_ramp_headroom_up_mw: float,
    gt_ramp_headroom_down_mw: float,
    tes_energy_mwh: float,
    tes_hot_k: float,
    abs_drive_margin_k: float,
) -> dict[str, float]:
    timestamp = pd.to_datetime(row["timestamp"])
    minute_of_day = timestamp.hour * 60 + timestamp.minute
    minute_of_week = timestamp.dayofweek * 24 * 60 + minute_of_day
    day_angle = 2.0 * math.pi * minute_of_day / (24.0 * 60.0)
    week_angle = 2.0 * math.pi * minute_of_week / (7.0 * 24.0 * 60.0)

    return {
        "p_dem_mw": float(row["p_dem_mw"]),
        "qh_dem_mw": float(row["qh_dem_mw"]),
        "qc_dem_mw": float(row["qc_dem_mw"]),
        "pv_mw": float(row["pv_mw"]),
        "wt_mw": float(row["wt_mw"]),
        "price_e": float(row["price_e"]),
        "price_gas": float(row["price_gas"]),
        "carbon_tax": float(row["carbon_tax"]),
        "t_amb_k": float(row["t_amb_k"]),
        "sp_pa": float(row["sp_pa"]),
        "rh_pct": float(row["rh_pct"]),
        "wind_speed": float(row["wind_speed"]),
        "wind_direction": float(row["wind_direction"]),
        "ghi_wm2": float(row["ghi_wm2"]),
        "dni_wm2": float(row["dni_wm2"]),
        "dhi_wm2": float(row["dhi_wm2"]),
        "soc_bes": float(bes_soc),
        "gt_on": float(1.0 if gt_prev_on else 0.0),
        "gt_state": float(gt_state),
        "p_gt_prev_mw": float(p_gt_prev_mw),
        "gt_ramp_headroom_up_mw": float(gt_ramp_headroom_up_mw),
        "gt_ramp_headroom_down_mw": float(gt_ramp_headroom_down_mw),
        "e_tes_mwh": float(tes_energy_mwh),
        "t_tes_hot_k": float(tes_hot_k),
        "abs_drive_margin_k": float(abs_drive_margin_k),
        "sin_t": float(math.sin(day_angle)),
        "cos_t": float(math.cos(day_angle)),
        "sin_week": float(math.sin(week_angle)),
        "cos_week": float(math.cos(week_angle)),
    }
