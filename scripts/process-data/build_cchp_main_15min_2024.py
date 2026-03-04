from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


def _has_feb29(year: int, tz: str) -> bool:
    try:
        _ = pd.Timestamp(year=year, month=2, day=29, tz=tz)
        return True
    except ValueError:
        return False


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _to_path_list(items: Iterable[str]) -> List[Path]:
    return [Path(s) for s in items]


def build_target_index_15min(year: int, tz: str, drop_feb29: bool) -> pd.DatetimeIndex:
    start = pd.Timestamp(f"{year}-01-01 00:00:00", tz=tz)
    end = pd.Timestamp(f"{year+1}-01-01 00:00:00", tz=tz)
    idx = pd.date_range(start=start, end=end, freq="15min", inclusive="left")

    if drop_feb29:
        if _has_feb29(year, tz):
            # Drop whole day if it exists.
            feb29_start = pd.Timestamp(f"{year}-02-29 00:00:00", tz=tz)
            feb29_end = pd.Timestamp(f"{year}-03-01 00:00:00", tz=tz)
            idx = idx[(idx < feb29_start) | (idx >= feb29_end)]

    expected = 96 * 365
    if len(idx) != expected:
        raise ValueError(f"target 15min index length must be {expected}, got {len(idx)}")
    return idx


def _drop_feb29(df: pd.DataFrame, ts_col: str, tz: str, year: int) -> pd.DataFrame:
    feb29 = f"{year}-02-29"
    mask = df[ts_col].dt.strftime("%Y-%m-%d") == feb29
    return df.loc[~mask].copy()


def load_industrial_park_2020_15min_mw(
    five_minute_dirs: List[Path],
    tz: str,
    year_load: int,
    drop_feb29: bool,
    target_peak_mw: float,
) -> Tuple[pd.Series, Dict[str, Any]]:
    files: List[Path] = []
    for d in five_minute_dirs:
        if not d.exists():
            raise FileNotFoundError(f"industrial park 5min dir not found: {d}")
        files.extend(sorted(d.glob("*.csv")))

    if not files:
        raise FileNotFoundError("no industrial park 5min CSV files found")

    dfs = []
    for p in files:
        df = pd.read_csv(p, usecols=["Time", "Power (kW)"])
        df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
        df = df.dropna(subset=["Time"]).copy()
        df = df.rename(columns={"Power (kW)": "power_kw"})
        dfs.append(df)

    raw = pd.concat(dfs, ignore_index=True)
    raw = raw.dropna(subset=["Time"]).copy()
    raw = raw[(raw["Time"].dt.year == year_load)].copy()

    # Localize as naive local time -> timezone aware (Asia/Shanghai).
    raw["Time"] = raw["Time"].dt.tz_localize(tz)

    if drop_feb29:
        raw = _drop_feb29(raw, "Time", tz=tz, year=year_load)

    grp = raw.groupby("Time", as_index=True)["power_kw"]
    total_5min_kw = grp.sum()
    count_per_ts = grp.size()

    # Reindex to a full 5min grid for robustness.
    start = pd.Timestamp(f"{year_load}-01-01 00:00:00", tz=tz)
    end = pd.Timestamp(f"{year_load+1}-01-01 00:00:00", tz=tz)
    grid_5min = pd.date_range(start=start, end=end, freq="5min", inclusive="left")
    if drop_feb29:
        feb29_start = pd.Timestamp(f"{year_load}-02-29 00:00:00", tz=tz)
        feb29_end = pd.Timestamp(f"{year_load}-03-01 00:00:00", tz=tz)
        grid_5min = grid_5min[(grid_5min < feb29_start) | (grid_5min >= feb29_end)]
    total_5min_kw = total_5min_kw.reindex(grid_5min)
    total_5min_kw = total_5min_kw.interpolate(method="time").ffill().bfill()

    # Power series: 5min -> 15min mean power.
    total_15min_kw = total_5min_kw.resample("15min").mean()
    # Resample creates bins over the full [min,max] time span, including the removed leap day.
    # Freeze: explicitly reindex to a 365-day 15min grid (Feb 29 removed) to get 35,040 rows.
    grid_15min = pd.date_range(
        start=pd.Timestamp(f"{year_load}-01-01 00:00:00", tz=tz),
        end=pd.Timestamp(f"{year_load+1}-01-01 00:00:00", tz=tz),
        freq="15min",
        inclusive="left",
    )
    if drop_feb29 and _has_feb29(year_load, tz):
        feb29_start = pd.Timestamp(f"{year_load}-02-29 00:00:00", tz=tz)
        feb29_end = pd.Timestamp(f"{year_load}-03-01 00:00:00", tz=tz)
        grid_15min = grid_15min[(grid_15min < feb29_start) | (grid_15min >= feb29_end)]
    total_15min_kw = total_15min_kw.reindex(grid_15min).interpolate(method="time").ffill().bfill()

    expected = 96 * 365
    if len(total_15min_kw) != expected:
        raise ValueError(f"industrial park 15min length must be {expected}, got {len(total_15min_kw)}")

    total_15min_mw = total_15min_kw / 1000.0
    peak_before = float(total_15min_mw.max())
    if peak_before <= 0:
        raise ValueError("industrial park peak power must be > 0")

    scale = target_peak_mw / peak_before
    total_15min_mw = total_15min_mw * scale

    meta = {
        "files_count": len(files),
        "unique_5min_timestamps": int(total_5min_kw.notna().sum()),
        "min_count_per_5min_timestamp": int(count_per_ts.min()),
        "max_count_per_5min_timestamp": int(count_per_ts.max()),
        "peak_mw_before_scale": peak_before,
        "peak_mw_after_scale": float(total_15min_mw.max()),
        "scale_to_target_peak": scale,
        "target_peak_mw": target_peak_mw,
        "drop_feb29": drop_feb29,
        "year_load": year_load,
    }
    return total_15min_mw, meta


def load_citylearn_vt_47_15min_mw(
    dataset_dir: Path,
    file_glob: str,
    tz: str,
    target_year: int,
    drop_feb29: bool,
    qh_peak_ratio_to_pdem_peak: float,
    qc_peak_ratio_to_pdem_peak: float,
    pdem_peak_mw: float,
) -> Tuple[pd.Series, pd.Series, Dict[str, Any]]:
    if not dataset_dir.exists():
        raise FileNotFoundError(f"CityLearn dataset dir not found: {dataset_dir}")

    files = sorted(dataset_dir.glob(file_glob))
    if not files:
        raise FileNotFoundError(f"no CityLearn files matched {file_glob} in {dataset_dir}")

    # CityLearn is 8760 rows (1h typical-year). We map sequentially onto target_year (drop Feb 29).
    start = pd.Timestamp(f"{target_year}-01-01 00:00:00", tz=tz)
    end = pd.Timestamp(f"{target_year+1}-01-01 00:00:00", tz=tz)
    idx_1h = pd.date_range(start=start, end=end, freq="1h", inclusive="left")
    if drop_feb29 and _has_feb29(target_year, tz):
        feb29_start = pd.Timestamp(f"{target_year}-02-29 00:00:00", tz=tz)
        feb29_end = pd.Timestamp(f"{target_year}-03-01 00:00:00", tz=tz)
        idx_1h = idx_1h[(idx_1h < feb29_start) | (idx_1h >= feb29_end)]

    if len(idx_1h) != 8760:
        raise ValueError(f"target 1h index length must be 8760, got {len(idx_1h)}")

    # Aggregate 47 buildings.
    qh_kwh = np.zeros(8760, dtype=np.float64)
    qc_kwh = np.zeros(8760, dtype=np.float64)

    used_files = 0
    for p in files:
        df = pd.read_csv(
            p,
            usecols=["cooling_demand", "heating_demand", "dhw_demand"],
        )
        if len(df) != 8760:
            raise ValueError(f"CityLearn file {p} rows must be 8760, got {len(df)}")
        df = df.fillna(0.0)
        qc_kwh += df["cooling_demand"].to_numpy(dtype=np.float64)
        qh_kwh += (df["heating_demand"] + df["dhw_demand"]).to_numpy(dtype=np.float64)
        used_files += 1

    # Convert kWh/step (1h) -> kW -> MW
    qc_mw_1h = pd.Series(qc_kwh / 1000.0, index=idx_1h, name="qc_dem_mw")
    qh_mw_1h = pd.Series(qh_kwh / 1000.0, index=idx_1h, name="qh_dem_mw")

    # Upsample to 15min by ZOH (repeat each hour 4 times).
    idx_15min = build_target_index_15min(target_year, tz, drop_feb29)
    qc_15 = pd.Series(np.repeat(qc_mw_1h.to_numpy(), 4), index=idx_15min, name="qc_dem_mw")
    qh_15 = pd.Series(np.repeat(qh_mw_1h.to_numpy(), 4), index=idx_15min, name="qh_dem_mw")

    # Scale to match electric demand magnitude.
    qc_peak_before = float(qc_15.max())
    qh_peak_before = float(qh_15.max())
    if qc_peak_before <= 0 or qh_peak_before <= 0:
        raise ValueError("CityLearn aggregated peaks must be > 0")

    qc_target_peak = qc_peak_ratio_to_pdem_peak * pdem_peak_mw
    qh_target_peak = qh_peak_ratio_to_pdem_peak * pdem_peak_mw

    s_qc = qc_target_peak / qc_peak_before
    s_qh = qh_target_peak / qh_peak_before

    qc_15 = qc_15 * s_qc
    qh_15 = qh_15 * s_qh

    meta = {
        "dataset_dir": str(dataset_dir).replace("\\", "/"),
        "file_glob": file_glob,
        "buildings_count": used_files,
        "assumption": "CityLearn resstock-*.csv row0 maps to target_year-01-01 00:00 (local time), sequential for 8760 hours.",
        "qh_definition": "heating_demand + dhw_demand (kWh/step) treated as heat demand, then /1000 -> MW at 1h steps.",
        "upsample_1h_to_15min": "zero-order hold (repeat each hour 4 times)",
        "scale_qh_peak_ratio_to_pdem_peak": qh_peak_ratio_to_pdem_peak,
        "scale_qc_peak_ratio_to_pdem_peak": qc_peak_ratio_to_pdem_peak,
        "pdem_peak_mw_used_for_scaling": pdem_peak_mw,
        "qh_peak_mw_before_scale": qh_peak_before,
        "qc_peak_mw_before_scale": qc_peak_before,
        "qh_scale_factor": s_qh,
        "qc_scale_factor": s_qc,
        "qh_peak_mw_after_scale": float(qh_15.max()),
        "qc_peak_mw_after_scale": float(qc_15.max()),
    }
    return qh_15, qc_15, meta


def load_beijing_weather_15min(
    source_csv: Path,
    tz: str,
    target_year: int,
    drop_feb29: bool,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if not source_csv.exists():
        raise FileNotFoundError(f"weather source CSV not found: {source_csv}")

    df = pd.read_csv(source_csv)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).copy()

    # Source contains utc_offset column (=8.0); treat datetime as local time already.
    # (If needed later, we can switch to a true UTC->local conversion; for now we keep KISS.)
    df["datetime"] = df["datetime"].dt.tz_localize(tz)
    df = df[(df["datetime"].dt.year == target_year)].copy()

    if drop_feb29:
        df = _drop_feb29(df, "datetime", tz=tz, year=target_year)

    keep_cols = {
        "temperature": "t_amb_c",
        "surface_pressure": "sp_pa",
        "relative_humidity": "rh_frac",
        "wind_speed": "wind_speed",
        "wind_direction": "wind_direction",
        "surface_solar_radiation": "ghi_wm2",
        "direct_normal_solar_radiation": "dni_wm2",
        "surface_diffuse_solar_radiation": "dhi_wm2",
    }

    out = df[["datetime", *keep_cols.keys()]].rename(columns=keep_cols)
    out = out.set_index("datetime").sort_index()

    # Convert to frozen naming/units.
    out["t_amb_k"] = out["t_amb_c"] + 273.15
    out["rh_pct"] = out["rh_frac"] * 100.0
    out = out.drop(columns=["t_amb_c", "rh_frac"])

    # Build full 1h grid and fill, then upsample to 15min using ZOH.
    start = pd.Timestamp(f"{target_year}-01-01 00:00:00", tz=tz)
    end = pd.Timestamp(f"{target_year+1}-01-01 00:00:00", tz=tz)
    idx_1h = pd.date_range(start=start, end=end, freq="1h", inclusive="left")
    if drop_feb29 and _has_feb29(target_year, tz):
        feb29_start = pd.Timestamp(f"{target_year}-02-29 00:00:00", tz=tz)
        feb29_end = pd.Timestamp(f"{target_year}-03-01 00:00:00", tz=tz)
        idx_1h = idx_1h[(idx_1h < feb29_start) | (idx_1h >= feb29_end)]

    out = out.reindex(idx_1h).ffill().bfill()
    out_15 = out.resample("15min").ffill()
    idx_15min = build_target_index_15min(target_year, tz=tz, drop_feb29=drop_feb29)
    out_15 = out_15.reindex(idx_15min).ffill().bfill()

    expected = 96 * 365
    if len(out_15) != expected:
        raise ValueError(f"weather 15min length must be {expected}, got {len(out_15)}")

    meta = {
        "source_csv": str(source_csv).replace("\\", "/"),
        "assumption": "Oikolab ERA5 merged CSV datetime treated as local time (utc_offset=8).",
        "upsample_1h_to_15min": "zero-order hold (ffill after resample)",
        "drop_feb29": drop_feb29,
        "target_year": target_year,
    }
    return out_15, meta


def _tou_level_for_timestamp(ts: pd.Series) -> pd.Series:
    # Input: timezone-aware timestamps in Asia/Shanghai.
    month = ts.dt.month
    hour = ts.dt.hour
    minute = ts.dt.minute
    hm = hour * 60 + minute

    # Base TOU levels.
    # valley: 23:00-07:00
    valley = (hm >= 23 * 60) | (hm < 7 * 60)
    # peak: 10:00-13:00 and 17:00-22:00
    peak = ((hm >= 10 * 60) & (hm < 13 * 60)) | ((hm >= 17 * 60) & (hm < 22 * 60))
    # flat: rest (7:00-10:00, 13:00-17:00, 22:00-23:00)
    level = np.where(valley, "valley", np.where(peak, "peak", "flat"))

    # Super-peak overrides.
    super_peak = (
        ((month.isin([7, 8])) & (((hm >= 11 * 60) & (hm < 13 * 60)) | ((hm >= 16 * 60) & (hm < 17 * 60))))
        | ((month.isin([1, 12])) & ((hm >= 18 * 60) & (hm < 21 * 60)))
    )
    level = np.where(super_peak, "super_peak", level)
    return pd.Series(level, index=ts.index)


def build_price_e_series(
    idx_15min: pd.DatetimeIndex,
    flat_prices_csv: Path,
    flat_price_selection: Dict[str, str],
    multipliers_csv: Path,
    tariff_for_multipliers: str,
) -> Tuple[pd.Series, Dict[str, Any]]:
    if not flat_prices_csv.exists():
        raise FileNotFoundError(f"flat prices CSV not found: {flat_prices_csv}")
    if not multipliers_csv.exists():
        raise FileNotFoundError(f"TOU multipliers CSV not found: {multipliers_csv}")

    flat = pd.read_csv(flat_prices_csv)
    sel = flat.copy()
    for k, v in flat_price_selection.items():
        sel = sel[sel[k] == v]
    if len(sel) != 1:
        raise ValueError(f"flat price selection must match exactly 1 row, got {len(sel)}")
    flat_cny_per_kwh = float(sel.iloc[0]["flat_price_cny_per_kwh"])

    mult = pd.read_csv(multipliers_csv)
    mult = mult[mult["tariff"] == tariff_for_multipliers].copy()
    if mult.empty:
        raise ValueError(f"no multipliers found for tariff={tariff_for_multipliers}")
    mult_map = dict(zip(mult["tou_level"], mult["multiplier_to_flat"]))

    ts = pd.Series(idx_15min, index=idx_15min)
    level = _tou_level_for_timestamp(ts)
    multiplier = level.map(mult_map).astype(float)
    price_cny_per_kwh = flat_cny_per_kwh * multiplier
    price_cny_per_mwh = price_cny_per_kwh * 1000.0

    meta = {
        "flat_price_cny_per_kwh": flat_cny_per_kwh,
        "flat_price_selection": flat_price_selection,
        "tariff_for_multipliers": tariff_for_multipliers,
        "multipliers_used": {k: float(v) for k, v in mult_map.items()},
        "output_unit": "CNY/MWh",
        "tou_level_counts": level.value_counts().to_dict(),
    }
    return pd.Series(price_cny_per_mwh.to_numpy(), index=idx_15min, name="price_e"), meta


def wind_power_from_speed(
    wind_speed_ms: pd.Series,
    wt_cap_mw: float,
    cut_in_ms: float,
    rated_ms: float,
    cut_out_ms: float,
) -> pd.Series:
    if wt_cap_mw <= 0:
        return pd.Series(np.zeros(len(wind_speed_ms)), index=wind_speed_ms.index, name="wt_mw")

    v = wind_speed_ms.to_numpy(dtype=np.float64)
    p = np.zeros_like(v)

    on = (v >= cut_in_ms) & (v < cut_out_ms)
    below_rated = on & (v < rated_ms)
    above_rated = on & (v >= rated_ms)

    # cubic between cut-in and rated
    p[below_rated] = ((v[below_rated] - cut_in_ms) / (rated_ms - cut_in_ms)) ** 3
    p[above_rated] = 1.0

    return pd.Series(p * wt_cap_mw, index=wind_speed_ms.index, name="wt_mw")


def pv_power_from_ghi(
    ghi_wm2: pd.Series,
    pv_cap_mw: float,
    pv_derate: float,
    ref_wm2: float,
) -> pd.Series:
    if pv_cap_mw <= 0:
        return pd.Series(np.zeros(len(ghi_wm2)), index=ghi_wm2.index, name="pv_mw")
    cf = (ghi_wm2 / ref_wm2).clip(lower=0.0, upper=1.0)
    return pd.Series((pv_cap_mw * pv_derate) * cf.to_numpy(dtype=np.float64), index=ghi_wm2.index, name="pv_mw")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build data/processed/cchp_main_15min_2024.csv (15min × 365d) from raw sources.")
    parser.add_argument("--config", type=str, default="scripts/process-data/cchp_main_15min_2024_config.json")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = _read_json(cfg_path)

    tz = cfg["timezone"]
    target_year = int(cfg["target_year"])
    drop_feb29 = bool(cfg["drop_feb29"])

    idx_15 = build_target_index_15min(target_year, tz=tz, drop_feb29=drop_feb29)

    # 1) Electric load from industrial park.
    ind_cfg = cfg["industrial_park"]
    p_dem_mw_2020, ind_meta = load_industrial_park_2020_15min_mw(
        five_minute_dirs=_to_path_list(ind_cfg["five_minute_dirs"]),
        tz=tz,
        year_load=int(ind_cfg["year_load"]),
        drop_feb29=True,
        target_peak_mw=float(ind_cfg["target_peak_mw"]),
    )
    # Map 2020 profile (365d after dropping Feb29) onto target index sequentially.
    p_dem_mw = pd.Series(p_dem_mw_2020.to_numpy(), index=idx_15, name="p_dem_mw")

    # 2) Heat/cool demands from CityLearn (47 buildings) scaled to pdem peak ratios.
    cl_cfg = cfg["citylearn"]
    qh_dem_mw, qc_dem_mw, cl_meta = load_citylearn_vt_47_15min_mw(
        dataset_dir=Path(cl_cfg["dataset_dir"]),
        file_glob=cl_cfg["file_glob"],
        tz=tz,
        target_year=target_year,
        drop_feb29=drop_feb29,
        qh_peak_ratio_to_pdem_peak=float(cl_cfg["scale_qh_peak_ratio_to_pdem_peak"]),
        qc_peak_ratio_to_pdem_peak=float(cl_cfg["scale_qc_peak_ratio_to_pdem_peak"]),
        pdem_peak_mw=float(p_dem_mw.max()),
    )

    # 3) Weather (Beijing ERA5 via Oikolab) -> 15min.
    w_cfg = cfg["weather"]
    weather_15, w_meta = load_beijing_weather_15min(
        source_csv=Path(w_cfg["source_csv"]),
        tz=tz,
        target_year=target_year,
        drop_feb29=drop_feb29,
    )
    weather_15 = weather_15.reindex(idx_15).ffill().bfill()

    # 4) Renewables (simple physics).
    r_cfg = cfg["renewables"]
    pv_mw = pv_power_from_ghi(
        ghi_wm2=weather_15["ghi_wm2"],
        pv_cap_mw=float(r_cfg["pv_cap_mw"]),
        pv_derate=float(r_cfg["pv_derate"]),
        ref_wm2=float(r_cfg["pv_irradiance_ref_wm2"]),
    )
    wt_mw = wind_power_from_speed(
        wind_speed_ms=weather_15["wind_speed"],
        wt_cap_mw=float(r_cfg["wt_cap_mw"]),
        cut_in_ms=float(r_cfg["wt_cut_in_ms"]),
        rated_ms=float(r_cfg["wt_rated_ms"]),
        cut_out_ms=float(r_cfg["wt_cut_out_ms"]),
    )

    # 5) Prices.
    pr_cfg = cfg["prices"]
    e_cfg = pr_cfg["electricity"]
    price_e, price_e_meta = build_price_e_series(
        idx_15min=idx_15,
        flat_prices_csv=Path(e_cfg["flat_prices_csv"]),
        flat_price_selection=e_cfg["flat_price_selection"],
        multipliers_csv=Path(e_cfg["tou_multipliers_csv"]),
        tariff_for_multipliers=e_cfg["tariff_for_multipliers"],
    )

    gas_cfg = pr_cfg["gas"]
    if gas_cfg.get("mode") != "constant_cny_per_gj":
        raise ValueError("only gas mode supported now: constant_cny_per_gj")
    price_gas = pd.Series(
        np.full(len(idx_15), float(gas_cfg["constant_cny_per_gj"]), dtype=np.float64),
        index=idx_15,
        name="price_gas",
    )

    carbon_cfg = pr_cfg["carbon"]
    carbon_tax = pd.Series(
        np.full(len(idx_15), float(carbon_cfg["carbon_tax_cny_per_tco2"]), dtype=np.float64),
        index=idx_15,
        name="carbon_tax",
    )

    # 6) Assemble (left join to timestamp index).
    out = pd.DataFrame(index=idx_15)
    out.index.name = "timestamp"
    out = out.join(p_dem_mw).join(qh_dem_mw).join(qc_dem_mw)
    out = out.join(pv_mw).join(wt_mw)
    out = out.join(weather_15[["t_amb_k", "sp_pa", "rh_pct", "wind_speed", "wind_direction", "ghi_wm2", "dni_wm2", "dhi_wm2"]])
    out = out.join(price_e).join(price_gas).join(carbon_tax)

    # Final missing value policy (frozen): weather ffill, loads time interpolation+edge fill.
    for c in ["t_amb_k", "sp_pa", "rh_pct", "wind_speed", "wind_direction", "ghi_wm2", "dni_wm2", "dhi_wm2"]:
        out[c] = out[c].ffill().bfill()

    for c in ["p_dem_mw", "qh_dem_mw", "qc_dem_mw"]:
        out[c] = out[c].interpolate(method="time").ffill().bfill()

    for c in ["pv_mw", "wt_mw"]:
        out[c] = out[c].fillna(0.0)

    for c in ["price_e", "price_gas", "carbon_tax"]:
        out[c] = out[c].ffill().bfill()

    # Write outputs.
    out_csv = Path(cfg["output"]["main_csv"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_to_write = out.reset_index()
    out_to_write.to_csv(out_csv, index=False)

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": str(cfg_path).replace("\\", "/"),
        "config": cfg,
        "shape": {"rows": int(len(out_to_write)), "cols": int(out_to_write.shape[1])},
        "columns": list(out_to_write.columns),
        "index": {
            "timezone": tz,
            "target_year": target_year,
            "drop_feb29": drop_feb29,
            "rows_expected": 96 * 365,
        },
        "industrial_park_meta": ind_meta,
        "citylearn_meta": cl_meta,
        "weather_meta": w_meta,
        "price_e_meta": price_e_meta,
    }
    _write_json(Path(cfg["output"]["manifest_json"]), manifest)

    print(f"Wrote {out_csv} rows={len(out_to_write)} cols={out_to_write.shape[1]}")
    print(f"Wrote {cfg['output']['manifest_json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
