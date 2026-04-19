import unittest
from types import SimpleNamespace

import torch

from src.cchp_physical_env.policy.pafc_td3 import (
    PAFCTD3Trainer,
    _gt_anchor_relax_signal_np,
    _gt_proxy_support_multiplier_np,
)


class PAFCGTEconomicProxyTest(unittest.TestCase):
    def test_support_multiplier_increases_with_heat_support_need(self) -> None:
        weak_heat = _gt_proxy_support_multiplier_np(abs_ready=0.9, heat_support_need=0.1)
        strong_heat = _gt_proxy_support_multiplier_np(abs_ready=0.9, heat_support_need=0.8)
        self.assertGreater(strong_heat, weak_heat)

    def test_anchor_relax_signal_prefers_profitable_undercommitted_gt(self) -> None:
        conservative = _gt_anchor_relax_signal_np(
            price_advantage=0.75,
            net_grid_need_ratio=0.55,
            undercommit_ratio=0.45,
            abs_ready=0.9,
            heat_support_need=0.4,
            projection_risk=0.0,
        )
        projected = _gt_anchor_relax_signal_np(
            price_advantage=0.75,
            net_grid_need_ratio=0.55,
            undercommit_ratio=0.45,
            abs_ready=0.9,
            heat_support_need=0.4,
            projection_risk=1.0,
        )
        self.assertGreater(conservative, 0.0)
        self.assertEqual(projected, 0.0)

    def test_grid_proxy_targets_above_min_output_in_profitable_absorbable_window(self) -> None:
        trainer = object.__new__(PAFCTD3Trainer)
        trainer.torch = torch
        trainer.action_index = {"u_gt": 0, "u_bes": 1, "u_ech": 2}
        trainer.observation_index = {
            "p_dem_mw": 0,
            "pv_mw": 1,
            "wt_mw": 2,
            "price_e": 3,
            "price_gas": 4,
            "t_amb_k": 5,
            "heat_backup_min_needed_mw": 6,
            "abs_drive_margin_k": 7,
            "qc_dem_mw": 8,
            "p_gt_prev_mw": 9,
        }
        trainer.env_config = SimpleNamespace(
            p_gt_cap_mw=12.0,
            gt_min_output_mw=1.0,
            q_boiler_cap_mw=10.0,
            q_abs_cool_cap_mw=4.5,
            q_ech_cap_mw=6.0,
            abs_gate_scale_k=2.0,
            p_bes_cap_mw=4.0,
            bes_eta_charge=0.95,
            bes_eta_discharge=0.95,
            cop_nominal=0.75,
            ech_cop_partload_min_fraction=0.72,
            gt_eta_min=0.26,
            gt_eta_max=0.36,
            gt_om_var_cost_per_mwh=10.0,
            gt_emission_ton_per_mwh_th=0.05,
            gt_start_cost=25.0,
            gt_cycle_cost=5.0,
            dt_hours=0.25,
        )
        trainer.bes_price_low_threshold = 256.0
        trainer.bes_price_high_threshold = 1539.0
        trainer._compute_gt_price_prior_terms = PAFCTD3Trainer._compute_gt_price_prior_terms.__get__(
            trainer,
            PAFCTD3Trainer,
        )
        trainer._compute_gt_grid_proxy_terms = PAFCTD3Trainer._compute_gt_grid_proxy_terms.__get__(
            trainer,
            PAFCTD3Trainer,
        )
        obs = torch.tensor(
            [[8.5, 0.0, 0.0, 1500.0, 80.0, 301.15, 0.2, 4.0, 1.0, 1.0]],
            dtype=torch.float32,
        )
        min_output_u = 2.0 * (1.0 / 12.0) - 1.0
        action_exec = torch.tensor([[min_output_u, 0.0, 0.0]], dtype=torch.float32)

        terms = trainer._compute_gt_grid_proxy_terms(
            obs_batch=obs,
            action_exec_batch=action_exec,
        )

        self.assertIsNotNone(terms)
        self.assertGreater(float(terms["gt_target_proxy_mw"][0, 0].detach().cpu().item()), 1.0)
        self.assertGreater(float(terms["undercommit_ratio"][0, 0].detach().cpu().item()), 0.0)
        self.assertGreater(float(terms["net_grid_absorb_ratio"][0, 0].detach().cpu().item()), 0.5)
        self.assertGreater(float(terms["support_multiplier"][0, 0].detach().cpu().item()), 0.0)

    def test_grid_proxy_does_not_push_gt_when_min_output_is_not_absorbable(self) -> None:
        trainer = object.__new__(PAFCTD3Trainer)
        trainer.torch = torch
        trainer.action_index = {"u_gt": 0, "u_bes": 1, "u_ech": 2}
        trainer.observation_index = {
            "p_dem_mw": 0,
            "pv_mw": 1,
            "wt_mw": 2,
            "price_e": 3,
            "price_gas": 4,
            "t_amb_k": 5,
            "heat_backup_min_needed_mw": 6,
            "abs_drive_margin_k": 7,
            "qc_dem_mw": 8,
            "p_gt_prev_mw": 9,
        }
        trainer.env_config = SimpleNamespace(
            p_gt_cap_mw=12.0,
            gt_min_output_mw=1.0,
            q_boiler_cap_mw=10.0,
            q_abs_cool_cap_mw=4.5,
            q_ech_cap_mw=6.0,
            abs_gate_scale_k=2.0,
            p_bes_cap_mw=4.0,
            bes_eta_charge=0.95,
            bes_eta_discharge=0.95,
            cop_nominal=0.75,
            ech_cop_partload_min_fraction=0.72,
            gt_eta_min=0.26,
            gt_eta_max=0.36,
            gt_om_var_cost_per_mwh=10.0,
            gt_emission_ton_per_mwh_th=0.05,
            gt_start_cost=25.0,
            gt_cycle_cost=5.0,
            dt_hours=0.25,
        )
        trainer.bes_price_low_threshold = 256.0
        trainer.bes_price_high_threshold = 1539.0
        trainer._compute_gt_price_prior_terms = PAFCTD3Trainer._compute_gt_price_prior_terms.__get__(
            trainer,
            PAFCTD3Trainer,
        )
        trainer._compute_gt_grid_proxy_terms = PAFCTD3Trainer._compute_gt_grid_proxy_terms.__get__(
            trainer,
            PAFCTD3Trainer,
        )
        obs = torch.tensor(
            [[1.5, 1.0, 0.5, 220.0, 80.0, 298.15, 0.0, 4.0, 0.2, 0.0]],
            dtype=torch.float32,
        )
        action_exec = torch.tensor([[-1.0, 0.0, 0.0]], dtype=torch.float32)

        terms = trainer._compute_gt_grid_proxy_terms(
            obs_batch=obs,
            action_exec_batch=action_exec,
        )

        self.assertIsNotNone(terms)
        self.assertAlmostEqual(float(terms["gt_target_proxy_mw"][0, 0].detach().cpu().item()), 0.0, places=6)
        self.assertAlmostEqual(float(terms["undercommit_ratio"][0, 0].detach().cpu().item()), 0.0, places=6)
        self.assertAlmostEqual(float(terms["mode_on"][0, 0].detach().cpu().item()), 0.0, places=6)
        self.assertLess(float(terms["net_grid_absorb_ratio"][0, 0].detach().cpu().item()), 0.1)


if __name__ == "__main__":
    unittest.main()
