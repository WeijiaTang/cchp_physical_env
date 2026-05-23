"""Generate sanity tables for the dispatch-grade thermal device models.

The script does not train or replay a controller. It only evaluates the
algebraic GT/HRSG, ABS and ECH curves at representative operating points so
the assumptions behind the annual benchmark can be inspected and reported.
"""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cchp_physical_env.core.config_loader import (
    build_env_config_from_overrides,
    load_env_overrides,
)
from src.cchp_physical_env.env.physics.tespy import (
    AbsChillerDesignPoint,
    AbsChillerNetwork,
    ElectricChillerNetwork,
    GTDesignPoint,
    GTNetwork,
    HRSGDesignPoint,
    HRSGNetwork,
)


CONFIG = ROOT / "src" / "cchp_physical_env" / "config" / "config.yaml"
OUT_DIR = ROOT / "results" / "tables" / "diagnostics"


def build_models():
    cfg = build_env_config_from_overrides(load_env_overrides(CONFIG))
    gt = GTNetwork(
        GTDesignPoint(
            p_gt_cap_mw=cfg.p_gt_cap_mw,
            gt_eta_min=cfg.gt_eta_min,
            gt_eta_max=cfg.gt_eta_max,
            gt_eta_curve_exp=cfg.gt_eta_curve_exp,
            gt_eta_frac_25=cfg.gt_eta_frac_25,
            gt_eta_frac_50=cfg.gt_eta_frac_50,
            gt_eta_frac_75=cfg.gt_eta_frac_75,
            gt_eta_frac_100=cfg.gt_eta_frac_100,
            gas_lhv_mj_per_kg=cfg.gas_lhv_mj_per_kg,
            gt_min_output_mw=cfg.gt_min_output_mw,
            m_exh_per_fuel_ratio=cfg.m_exh_per_fuel_ratio,
            t_exh_offset_k=cfg.t_exh_offset_k,
            t_exh_slope_k_per_mw=cfg.t_exh_slope_k_per_mw,
            t_exh_load_curve_exp=cfg.gt_t_exh_load_curve_exp,
            t_exh_min_k=cfg.gt_t_exh_min_k,
            t_exh_max_k=cfg.gt_t_exh_max_k,
            gt_exh_temp_k_25=cfg.gt_exh_temp_k_25,
            gt_exh_temp_k_50=cfg.gt_exh_temp_k_50,
            gt_exh_temp_k_75=cfg.gt_exh_temp_k_75,
            gt_exh_temp_k_100=cfg.gt_exh_temp_k_100,
            gt_exh_flow_frac_25=cfg.gt_exh_flow_frac_25,
            gt_exh_flow_frac_50=cfg.gt_exh_flow_frac_50,
            gt_exh_flow_frac_75=cfg.gt_exh_flow_frac_75,
            gt_exh_flow_frac_100=cfg.gt_exh_flow_frac_100,
        ),
        build_tespy_topology=False,
    )
    hrsg = HRSGNetwork(
        HRSGDesignPoint(
            ua_mw_per_k=cfg.ua_mw_per_k,
            m_water_kg_per_s=cfg.hrsg_water_mass_flow_kg_per_s,
            t_water_in_k=cfg.hrsg_water_inlet_k,
            cp_exh_kj_per_kgk=cfg.hrsg_cp_exh_kj_per_kgk,
            cp_water_kj_per_kgk=cfg.hrsg_cp_water_kj_per_kgk,
            t_w_out_max_k=cfg.hrsg_t_w_out_max_k,
            t_exh_out_min_k=cfg.hrsg_t_exh_out_min_k,
            m_exh_ref_kg_per_s=cfg.hrsg_m_exh_ref_kg_per_s,
            k_a_flow_exponent=cfg.hrsg_k_a_flow_exponent,
            water_flow_min_fraction=cfg.hrsg_water_flow_min_fraction,
            water_flow_max_fraction=cfg.hrsg_water_flow_max_fraction,
            water_flow_exponent=cfg.hrsg_water_flow_exponent,
            pinch_min_k=cfg.hrsg_pinch_min_k,
        ),
        build_tespy_topology=False,
    )
    abs_chiller = AbsChillerNetwork(
        AbsChillerDesignPoint(
            q_drive_cap_mw=cfg.q_abs_drive_cap_mw,
            q_cool_cap_mw=cfg.q_abs_cool_cap_mw,
            t_drive_min_k=cfg.abs_t_drive_min_k,
            t_drive_ref_k=cfg.abs_t_drive_ref_k,
            cop_nominal=cfg.cop_nominal,
            cop_min_fraction=cfg.abs_cop_min_fraction,
            t_cooling_water_ref_k=cfg.abs_t_cooling_water_ref_k,
            t_evap_ref_k=cfg.abs_t_evap_ref_k,
            cooling_water_approach_k=cfg.abs_cooling_water_approach_k,
            cooling_tower_wetbulb_depression_k=cfg.abs_cooling_tower_wetbulb_depression_k,
            cooling_tower_range_design_k=cfg.abs_cooling_tower_range_design_k,
            cooling_tower_load_exponent=cfg.abs_cooling_tower_load_exponent,
            evap_temp_k=cfg.abs_evap_temp_k,
            chilled_water_supply_k=cfg.chilled_water_supply_k,
            chilled_water_return_k=cfg.chilled_water_return_k,
            cop_cooling_water_slope_per_k=cfg.abs_cop_cooling_water_slope_per_k,
            cop_evap_slope_per_k=cfg.abs_cop_evap_slope_per_k,
            cop_partload_min_fraction=cfg.abs_cop_partload_min_fraction,
            cop_partload_curve_exp=cfg.abs_cop_partload_curve_exp,
        )
    )
    ech = ElectricChillerNetwork(
        q_ech_cap_mw=cfg.q_ech_cap_mw,
        cop_nominal=cfg.ech_cop_nominal,
        cop_floor=cfg.ech_cop_floor,
        cop_temp_slope_per_k=cfg.ech_cop_temp_slope_per_k,
        cop_ref_temp_k=cfg.ech_cop_ref_temp_k,
        cop_partload_min_fraction=cfg.ech_cop_partload_min_fraction,
        cop_partload_curve_exp=cfg.ech_cop_partload_curve_exp,
        condenser_water_approach_k=cfg.ech_condenser_water_approach_k,
        cooling_tower_wetbulb_depression_k=cfg.ech_cooling_tower_wetbulb_depression_k,
        cooling_tower_range_design_k=cfg.ech_cooling_tower_range_design_k,
        cooling_tower_load_exponent=cfg.ech_cooling_tower_load_exponent,
        chilled_water_supply_k=cfg.chilled_water_supply_k,
        chilled_water_return_k=cfg.chilled_water_return_k,
        chilled_water_supply_ref_k=cfg.chilled_water_supply_k,
    )
    return cfg, gt, hrsg, abs_chiller, ech


def main() -> None:
    cfg, gt, hrsg, abs_chiller, ech = build_models()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    gt_hrsg_rows = []
    for load in [0.0, 0.25, 0.50, 0.75, 1.00]:
        gt_result = gt.solve_offdesign(
            p_gt_request_mw=load * cfg.p_gt_cap_mw,
            t_amb_k=298.15,
        )
        hrsg_result = hrsg.solve(
            m_exh_kg_per_s=gt_result.m_exh_kg_per_s,
            t_exh_in_k=gt_result.t_exh_k,
        )
        gt_hrsg_rows.append(
            {
                "gt_load_fraction": load,
                "p_gt_mw": gt_result.p_gt_mw,
                "eta_gt": gt_result.eta_gt,
                "fuel_input_mw": gt_result.fuel_input_mw,
                "m_exh_kg_per_s": gt_result.m_exh_kg_per_s,
                "t_exh_in_k": gt_result.t_exh_k,
                "q_hrsg_rec_mw": hrsg_result.q_rec_mw,
                "t_exh_out_k": hrsg_result.t_exh_out_k,
                "t_water_out_k": hrsg_result.t_water_out_k,
                "epsilon": hrsg_result.epsilon,
                "ua_effective_mw_per_k": hrsg_result.ua_effective_mw_per_k,
                "hrsg_pinch_limited": bool(hrsg_result.violation_flags.get("hrsg_pinch_limited", False)),
            }
        )
    pd.DataFrame(gt_hrsg_rows).to_csv(OUT_DIR / "gt_hrsg_sanity.csv", index=False)

    abs_rows = []
    for t_hot in [cfg.abs_t_drive_min_k - 1.0, cfg.abs_t_drive_min_k + 5.0, cfg.abs_t_drive_ref_k]:
        for t_cw in [cfg.abs_t_cooling_water_ref_k, cfg.abs_t_cooling_water_ref_k + 10.0]:
            for plr in [0.25, 0.75, 1.00]:
                abs_rows.append(
                    {
                        "t_hot_k": t_hot,
                        "t_cooling_water_k": t_cw,
                        "t_chw_supply_k": cfg.chilled_water_supply_k,
                        "t_chw_return_k": cfg.chilled_water_return_k,
                        "t_evap_k": cfg.abs_evap_temp_k,
                        "plr": plr,
                        "cop_abs": abs_chiller.estimate_cop(
                            t_hot_k=t_hot,
                            t_cooling_water_k=t_cw,
                            t_evap_k=cfg.abs_evap_temp_k,
                            plr=plr,
                        ),
                    }
                )
    pd.DataFrame(abs_rows).to_csv(OUT_DIR / "abs_cop_sanity.csv", index=False)

    ech_rows = []
    for t_amb in [288.15, 298.15, 308.15, 318.15]:
        for plr in [0.25, 0.75, 1.00]:
            q_cool = plr * cfg.q_ech_cap_mw
            cop_guess = ech.estimate_cop(t_amb_k=t_amb, plr=plr)
            t_cond = ech.estimate_condenser_water_temp(
                t_amb,
                heat_rejection_mw=q_cool + q_cool / max(1e-6, cop_guess),
            )
            ech_rows.append(
                {
                    "t_amb_k": t_amb,
                    "t_cond_in_k": t_cond,
                    "t_chw_supply_k": cfg.chilled_water_supply_k,
                    "t_chw_return_k": cfg.chilled_water_return_k,
                    "plr": plr,
                    "cop_ech": ech.estimate_cop(t_amb_k=t_amb, plr=plr, t_cond_in_k=t_cond),
                }
            )
    pd.DataFrame(ech_rows).to_csv(OUT_DIR / "ech_cop_sanity.csv", index=False)

    # Traceable engineering check points used to explain the dispatch surrogate.
    check_rows = []
    gt_targets = {
        0.25: {"eta": cfg.gt_eta_max * cfg.gt_eta_frac_25, "t_exh_k": cfg.gt_exh_temp_k_25},
        0.50: {"eta": cfg.gt_eta_max * cfg.gt_eta_frac_50, "t_exh_k": cfg.gt_exh_temp_k_50},
        0.75: {"eta": cfg.gt_eta_max * cfg.gt_eta_frac_75, "t_exh_k": cfg.gt_exh_temp_k_75},
        1.00: {"eta": cfg.gt_eta_max * cfg.gt_eta_frac_100, "t_exh_k": cfg.gt_exh_temp_k_100},
    }
    for load, target in gt_targets.items():
        gt_result = gt.solve_offdesign(p_gt_request_mw=load * cfg.p_gt_cap_mw, t_amb_k=288.15)
        check_rows.append(
            {
                "device": "GT",
                "point": f"{int(load * 100)}% load eta",
                "model_value": gt_result.eta_gt,
                "reference_value": target["eta"],
                "abs_error": abs(gt_result.eta_gt - target["eta"]),
                "unit": "-",
            }
        )
        check_rows.append(
            {
                "device": "GT",
                "point": f"{int(load * 100)}% load exhaust temperature",
                "model_value": gt_result.t_exh_k,
                "reference_value": target["t_exh_k"],
                "abs_error": abs(gt_result.t_exh_k - target["t_exh_k"]),
                "unit": "K",
            }
        )
    abs_ref = abs_chiller.estimate_cop(
        t_hot_k=cfg.abs_t_drive_ref_k,
        t_cooling_water_k=cfg.abs_t_cooling_water_ref_k,
        t_evap_k=cfg.abs_evap_temp_k,
        plr=1.0,
    )
    check_rows.append(
        {
            "device": "ABS",
            "point": "rated hot/chilled/cooling water",
            "model_value": abs_ref,
            "reference_value": cfg.cop_nominal,
            "abs_error": abs(abs_ref - cfg.cop_nominal),
            "unit": "-",
        }
    )
    ech_ref = ech.estimate_cop(
        t_amb_k=298.15,
        plr=1.0,
        t_cond_in_k=cfg.ech_cop_ref_temp_k,
        t_chw_supply_k=cfg.chilled_water_supply_k,
    )
    check_rows.append(
        {
            "device": "ECH",
            "point": "rated condenser/chilled water",
            "model_value": ech_ref,
            "reference_value": cfg.ech_cop_nominal,
            "abs_error": abs(ech_ref - cfg.ech_cop_nominal),
            "unit": "-",
        }
    )
    pd.DataFrame(check_rows).to_csv(OUT_DIR / "thermal_device_calibration_error.csv", index=False)
    print(f"Wrote sanity tables to {OUT_DIR}")


if __name__ == "__main__":
    main()
