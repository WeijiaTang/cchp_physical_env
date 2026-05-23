from __future__ import annotations

from pathlib import Path

import pytest

from src.cchp_physical_env.core.config_loader import (
    build_env_config_from_overrides,
    load_env_overrides,
)
from src.cchp_physical_env.env.physics.tespy.abs_chiller_network import (
    AbsChillerDesignPoint,
    AbsChillerNetwork,
)
from src.cchp_physical_env.env.physics.tespy.gt_network import GTDesignPoint, GTNetwork
from src.cchp_physical_env.env.physics.tespy.hrsg_network import HRSGDesignPoint, HRSGNetwork
from src.cchp_physical_env.env.physics.tespy.thermal_networks import ElectricChillerNetwork


def test_abs_cop_responds_to_drive_cooling_water_evap_and_partload() -> None:
    chiller = AbsChillerNetwork(
        AbsChillerDesignPoint(
            t_drive_min_k=348.15,
            t_drive_ref_k=368.15,
            cop_nominal=0.75,
            cop_min_fraction=0.50,
        )
    )

    low_drive = chiller.estimate_cop(
        t_hot_k=353.15,
        t_cooling_water_k=303.15,
        t_evap_k=280.15,
        plr=1.0,
    )
    high_drive = chiller.estimate_cop(
        t_hot_k=368.15,
        t_cooling_water_k=303.15,
        t_evap_k=280.15,
        plr=1.0,
    )
    hot_cooling_water = chiller.estimate_cop(
        t_hot_k=368.15,
        t_cooling_water_k=313.15,
        t_evap_k=280.15,
        plr=1.0,
    )
    low_partload = chiller.estimate_cop(
        t_hot_k=368.15,
        t_cooling_water_k=303.15,
        t_evap_k=280.15,
        plr=0.2,
    )

    assert high_drive > low_drive > 0.0
    assert high_drive > hot_cooling_water
    assert high_drive > low_partload
    assert chiller.estimate_cop(t_hot_k=347.15) == 0.0


def test_electric_chiller_cop_responds_to_ambient_and_partload() -> None:
    chiller = ElectricChillerNetwork(
        q_ech_cap_mw=6.0,
        cop_nominal=5.2,
        cop_floor=3.0,
        cop_temp_slope_per_k=0.03,
        cop_ref_temp_k=303.15,
        cop_partload_min_fraction=0.72,
        cop_partload_curve_exp=1.15,
    )

    cool_day = chiller.estimate_cop(t_amb_k=293.15, plr=1.0)
    hot_day = chiller.estimate_cop(t_amb_k=313.15, plr=1.0)
    low_partload = chiller.estimate_cop(t_amb_k=293.15, plr=0.2)

    assert cool_day > hot_day >= chiller.cop_floor
    assert cool_day > low_partload >= chiller.cop_floor


def test_cooling_tower_temperatures_increase_with_rejection_load() -> None:
    abs_chiller = AbsChillerNetwork(AbsChillerDesignPoint())
    ech = ElectricChillerNetwork()

    abs_idle = abs_chiller.estimate_cooling_water_temp(303.15, heat_rejection_mw=0.0)
    abs_loaded = abs_chiller.estimate_cooling_water_temp(303.15, heat_rejection_mw=8.0)
    ech_idle = ech.estimate_condenser_water_temp(303.15, heat_rejection_mw=0.0)
    ech_loaded = ech.estimate_condenser_water_temp(303.15, heat_rejection_mw=7.0)

    assert abs_loaded > abs_idle
    assert ech_loaded > ech_idle


def test_gt_hrsg_full_load_sanity_and_temperature_bounds() -> None:
    gt = GTNetwork(
        GTDesignPoint(
            p_gt_cap_mw=12.0,
            gt_eta_min=0.20,
            gt_eta_max=0.36,
            gas_lhv_mj_per_kg=50.0,
            m_exh_per_fuel_ratio=65.0,
            t_exh_offset_k=220.0,
            t_exh_slope_k_per_mw=0.0,
            t_exh_min_k=620.0,
            t_exh_max_k=850.0,
        ),
        build_tespy_topology=False,
    )
    hrsg = HRSGNetwork(
        HRSGDesignPoint(
            ua_mw_per_k=0.06,
            m_water_kg_per_s=45.0,
            t_water_in_k=333.15,
            cp_exh_kj_per_kgk=1.10,
            cp_water_kj_per_kgk=4.18,
            t_w_out_max_k=393.15,
            t_exh_out_min_k=380.15,
            m_exh_ref_kg_per_s=45.0,
            k_a_flow_exponent=0.8,
        ),
        build_tespy_topology=False,
    )

    gt_result = gt.solve_offdesign(p_gt_request_mw=12.0, t_amb_k=298.15)
    hrsg_result = hrsg.solve(
        m_exh_kg_per_s=gt_result.m_exh_kg_per_s,
        t_exh_in_k=gt_result.t_exh_k,
    )

    assert gt_result.p_gt_mw == pytest.approx(12.0)
    assert 0.35 <= gt_result.eta_gt <= 0.37
    assert gt_result.m_exh_kg_per_s == pytest.approx(43.33, rel=0.02)
    assert 780.0 <= gt_result.t_exh_k <= 805.0
    assert hrsg_result.q_rec_mw > 0.0
    assert hrsg_result.t_exh_out_k >= hrsg.design.t_exh_out_min_k - 1e-9
    assert hrsg_result.t_water_out_k <= hrsg.design.t_w_out_max_k + 1e-9


def test_released_yaml_exposes_thermal_model_parameters() -> None:
    config_path = Path("src/cchp_physical_env/config/config.yaml")
    cfg = build_env_config_from_overrides(load_env_overrides(config_path))

    assert cfg.abs_cop_partload_min_fraction == pytest.approx(0.85)
    assert cfg.abs_t_drive_min_k == pytest.approx(358.15)
    assert cfg.m_exh_per_fuel_ratio == pytest.approx(65.0)
    assert cfg.gt_exh_temp_k_100 == pytest.approx(793.15)
    assert cfg.hrsg_cp_exh_kj_per_kgk == pytest.approx(1.10)
    assert cfg.hrsg_m_exh_ref_kg_per_s == pytest.approx(45.0)
    assert cfg.hrsg_pinch_min_k == pytest.approx(15.0)
    assert cfg.ech_cop_nominal == pytest.approx(5.2)
