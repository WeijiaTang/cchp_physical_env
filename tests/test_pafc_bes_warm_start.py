import unittest
from types import SimpleNamespace

import numpy as np
import torch

from src.cchp_physical_env.policy.pafc_td3 import (
    PAFCTD3Trainer,
    _allocate_bes_source_counts,
    _allocate_bes_teacher_target_counts,
    _allocate_bes_warm_start_mode_counts,
    _bes_full_year_selection_priority_np,
    _bes_price_prior_target_np,
    _bes_warm_start_cooling_guard_active_np,
    _select_bes_full_year_target_np,
    _select_temporal_priority_indices,
    _select_temporal_priority_indices_by_season,
)


class PAFCBESWarmStartTest(unittest.TestCase):
    def test_allocate_bes_mode_counts_prefers_active_modes(self) -> None:
        counts = _allocate_bes_warm_start_mode_counts(
            requested_total=10,
            charge_available=6,
            discharge_available=2,
            idle_available=20,
        )
        self.assertEqual(sum(counts.values()), 10)
        self.assertEqual(counts["discharge"], 2)
        self.assertGreaterEqual(counts["charge"], counts["idle"])

    def test_allocate_bes_source_counts_preserves_economic_share_when_available(self) -> None:
        counts = _allocate_bes_source_counts(
            requested_total=10,
            economic_available=9,
            other_available=10,
            economic_min_share=0.70,
        )
        self.assertEqual(counts, {"economic": 7, "other": 3})

    def test_allocate_bes_teacher_target_counts_preserves_teacher_share_when_available(self) -> None:
        counts = _allocate_bes_teacher_target_counts(
            requested_total=10,
            teacher_available=8,
            other_available=10,
            teacher_min_share=0.60,
        )
        self.assertEqual(counts, {"teacher": 6, "other": 4})

    def test_temporal_priority_selection_keeps_time_coverage(self) -> None:
        selected = _select_temporal_priority_indices(
            indices=[0, 1, 2, 3, 4, 5],
            priorities=[0.1, 0.9, 0.2, 0.3, 0.8, 0.1],
            target_count=3,
        )
        self.assertEqual(selected, [1, 3, 4])

    def test_temporal_priority_selection_by_season_keeps_sparse_winter_summer_mix(self) -> None:
        selected = _select_temporal_priority_indices_by_season(
            indices=[0, 1, 4, 5],
            priorities=[0.2, 0.9, 0.3, 1.1, 0.8, 1.4],
            season_by_index={
                0: "winter",
                1: "winter",
                4: "summer",
                5: "summer",
            },
            target_count=2,
        )
        self.assertEqual(len(selected), 2)
        self.assertEqual({0, 1} & set(selected), {1})
        self.assertEqual({4, 5} & set(selected), {5})

    def test_bes_warm_start_cooling_guard_only_triggers_on_joint_risk(self) -> None:
        self.assertTrue(
            _bes_warm_start_cooling_guard_active_np(
                abs_drive_margin_k=0.8,
                qc_dem_mw=7.0,
                q_total_cooling_cap_mw=12.0,
                abs_margin_guard_k=1.25,
                qc_ratio_guard=0.55,
            )
        )
        self.assertFalse(
            _bes_warm_start_cooling_guard_active_np(
                abs_drive_margin_k=1.8,
                qc_dem_mw=7.0,
                q_total_cooling_cap_mw=12.0,
                abs_margin_guard_k=1.25,
                qc_ratio_guard=0.55,
            )
        )

    def test_bes_distill_loss_tracks_raw_action_not_exec_projection(self) -> None:
        trainer = object.__new__(PAFCTD3Trainer)
        trainer.torch = torch
        trainer.config = SimpleNamespace(
            economic_bes_distill_coef=1.0,
            economic_bes_charge_soc_ceiling=0.75,
            economic_bes_discharge_soc_floor=0.35,
            economic_bes_prior_u=0.35,
            economic_bes_charge_u_scale=1.0,
            economic_bes_discharge_u_scale=1.0,
            economic_bes_charge_weight=1.0,
            economic_bes_discharge_weight=1.0,
            economic_bes_charge_pressure_bonus=0.0,
        )
        trainer.env_config = SimpleNamespace(
            q_boiler_cap_mw=10.0,
            abs_gate_scale_k=2.0,
            bes_soc_min=0.10,
            bes_soc_max=0.95,
        )
        trainer.observation_index = {"price_e": 0, "soc_bes": 1}
        trainer.action_index = {"u_bes": 0}
        trainer.bes_price_low_threshold = 600.0
        trainer.bes_price_high_threshold = 1200.0

        prior = _bes_price_prior_target_np(
            price_e=1400.0,
            soc_bes=0.80,
            price_low_threshold=trainer.bes_price_low_threshold,
            price_high_threshold=trainer.bes_price_high_threshold,
            charge_soc_ceiling=trainer.config.economic_bes_charge_soc_ceiling,
            discharge_soc_floor=trainer.config.economic_bes_discharge_soc_floor,
            bes_soc_min=0.10,
            bes_soc_max=0.95,
            charge_u=trainer.config.economic_bes_prior_u,
            discharge_u=trainer.config.economic_bes_prior_u,
        )
        obs = torch.tensor([[1400.0, 0.80]], dtype=torch.float32)
        target_raw = torch.tensor([[float(prior["target_u_bes"])]], dtype=torch.float32)
        clipped_exec = torch.zeros((1, 1), dtype=torch.float32)
        gap = torch.zeros((1, 3), dtype=torch.float32)

        matched_loss, matched_weight = PAFCTD3Trainer._compute_bes_prior_distill_loss(
            trainer,
            obs_batch=obs,
            action_raw_batch=target_raw,
            action_exec_batch=clipped_exec,
            gap_batch=gap,
        )
        wrong_loss, _ = PAFCTD3Trainer._compute_bes_prior_distill_loss(
            trainer,
            obs_batch=obs,
            action_raw_batch=-target_raw,
            action_exec_batch=clipped_exec,
            gap_batch=gap,
        )

        self.assertGreater(float(matched_weight.item()), 0.0)
        self.assertAlmostEqual(float(matched_loss.item()), 0.0, places=6)
        self.assertGreater(float(wrong_loss.item()), 0.05)

    def test_bes_prior_allows_stronger_charge_targets_and_weights(self) -> None:
        charge_prior = _bes_price_prior_target_np(
            price_e=450.0,
            soc_bes=0.20,
            price_low_threshold=600.0,
            price_high_threshold=1200.0,
            charge_soc_ceiling=0.75,
            discharge_soc_floor=0.35,
            bes_soc_min=0.10,
            bes_soc_max=0.95,
            charge_u=0.56,
            discharge_u=0.35,
        )
        discharge_prior = _bes_price_prior_target_np(
            price_e=1400.0,
            soc_bes=0.80,
            price_low_threshold=600.0,
            price_high_threshold=1200.0,
            charge_soc_ceiling=0.75,
            discharge_soc_floor=0.35,
            bes_soc_min=0.10,
            bes_soc_max=0.95,
            charge_u=0.56,
            discharge_u=0.35,
        )

        self.assertEqual(str(charge_prior["mode"]), "charge")
        self.assertEqual(str(discharge_prior["mode"]), "discharge")
        self.assertGreater(abs(float(charge_prior["target_u_bes"])), abs(float(discharge_prior["target_u_bes"])))

        trainer = object.__new__(PAFCTD3Trainer)
        trainer.torch = torch
        trainer.config = SimpleNamespace(
            economic_bes_distill_coef=1.0,
            economic_bes_charge_soc_ceiling=0.75,
            economic_bes_discharge_soc_floor=0.35,
            economic_bes_prior_u=0.35,
            economic_bes_charge_u_scale=1.6,
            economic_bes_discharge_u_scale=1.0,
            economic_bes_charge_weight=2.0,
            economic_bes_discharge_weight=1.0,
            economic_bes_charge_pressure_bonus=1.0,
        )
        trainer.env_config = SimpleNamespace(
            q_boiler_cap_mw=10.0,
            abs_gate_scale_k=2.0,
            bes_soc_min=0.10,
            bes_soc_max=0.95,
        )
        trainer.observation_index = {"price_e": 0, "soc_bes": 1}
        trainer.action_index = {"u_bes": 0}
        trainer.bes_price_low_threshold = 600.0
        trainer.bes_price_high_threshold = 1200.0

        charge_obs = torch.tensor([[450.0, 0.20]], dtype=torch.float32)
        discharge_obs = torch.tensor([[1400.0, 0.80]], dtype=torch.float32)
        charge_raw = torch.tensor([[float(charge_prior["target_u_bes"])]], dtype=torch.float32)
        discharge_raw = torch.tensor([[float(discharge_prior["target_u_bes"])]], dtype=torch.float32)
        zero_exec = torch.zeros((1, 1), dtype=torch.float32)
        zero_gap = torch.zeros((1, 3), dtype=torch.float32)

        _, charge_weight = PAFCTD3Trainer._compute_bes_prior_distill_loss(
            trainer,
            obs_batch=charge_obs,
            action_raw_batch=charge_raw,
            action_exec_batch=zero_exec,
            gap_batch=zero_gap,
        )
        _, discharge_weight = PAFCTD3Trainer._compute_bes_prior_distill_loss(
            trainer,
            obs_batch=discharge_obs,
            action_raw_batch=discharge_raw,
            action_exec_batch=zero_exec,
            gap_batch=zero_gap,
        )
        self.assertGreater(float(charge_weight.item()), float(discharge_weight.item()))

    def test_teacher_bes_target_gets_strong_warm_start_weight(self) -> None:
        trainer = object.__new__(PAFCTD3Trainer)
        trainer.config = SimpleNamespace(
            economic_teacher_warm_start_weight=4.0,
            economic_bes_prior_u=0.35,
            economic_bes_charge_u_scale=1.6,
            economic_bes_discharge_u_scale=1.0,
            economic_bes_charge_weight=2.0,
            economic_bes_discharge_weight=1.0,
            economic_bes_charge_pressure_bonus=1.0,
            economic_bes_charge_soc_ceiling=0.75,
            economic_bes_discharge_soc_floor=0.35,
        )
        trainer.env_config = SimpleNamespace(
            bes_soc_min=0.10,
            bes_soc_max=0.95,
        )
        trainer.observation_index = {"price_e": 0, "soc_bes": 1}
        trainer.action_index = {
            "u_gt": 0,
            "u_bes": 1,
            "u_boiler": 2,
            "u_abs": 3,
            "u_ech": 4,
            "u_tes": 5,
        }
        trainer.action_keys = ("u_gt", "u_bes", "u_boiler", "u_abs", "u_ech", "u_tes")
        trainer.bes_price_low_threshold = 600.0
        trainer.bes_price_high_threshold = 1200.0
        trainer.economic_teacher_action_weight_np = np.asarray(
            [0.75, 1.50, 0.0, 0.0, 0.0, 0.50],
            dtype=np.float32,
        )

        warm_targets, sample_weight_boost, summary = PAFCTD3Trainer._build_actor_warm_start_targets(
            trainer,
            observations=np.asarray([[450.0, 0.20]], dtype=np.float32),
            action_exec_targets=np.zeros((1, 6), dtype=np.float32),
            teacher_action_exec=np.asarray([[0.0, -0.56, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            teacher_action_mask=np.asarray([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            teacher_available=np.asarray([[1.0]], dtype=np.float32),
        )

        self.assertAlmostEqual(float(warm_targets[0, 1]), -0.56, places=6)
        self.assertAlmostEqual(float(sample_weight_boost[0, 0]), 7.0, places=6)
        self.assertAlmostEqual(float(summary["teacher_sample_weight_boost_mean"]), 6.0, places=6)
        self.assertAlmostEqual(float(summary["teacher_sample_weight_boost_max"]), 6.0, places=6)

    def test_economic_teacher_can_override_idle_full_year_bes_target(self) -> None:
        selected = _select_bes_full_year_target_np(
            source_label="economic",
            prior_target_u_bes=0.0,
            prior_opportunity=0.0,
            teacher_u_bes=-0.42,
        )
        self.assertTrue(bool(selected["used_teacher"]))
        self.assertEqual(str(selected["mode"]), "charge")
        self.assertAlmostEqual(float(selected["target_u_bes"]), -0.42, places=6)

    def test_economic_teacher_does_not_flip_price_consistent_direction(self) -> None:
        selected = _select_bes_full_year_target_np(
            source_label="economic",
            prior_target_u_bes=-0.30,
            prior_opportunity=0.35,
            teacher_u_bes=0.55,
        )
        self.assertFalse(bool(selected["used_teacher"]))
        self.assertEqual(str(selected["mode"]), "charge")
        self.assertAlmostEqual(float(selected["target_u_bes"]), -0.30, places=6)

    def test_economic_teacher_rows_get_higher_selection_priority(self) -> None:
        baseline = _bes_full_year_selection_priority_np(
            base_priority=0.40,
            source_label="safe",
            used_teacher=False,
            teacher_priority_boost=0.75,
            economic_source_priority_bonus=0.10,
        )
        boosted = _bes_full_year_selection_priority_np(
            base_priority=0.40,
            source_label="economic",
            used_teacher=True,
            teacher_priority_boost=0.75,
            economic_source_priority_bonus=0.10,
        )
        self.assertAlmostEqual(float(baseline), 0.40, places=6)
        self.assertGreater(float(boosted), float(baseline))

    def test_teacher_bes_samples_preserve_stronger_online_anchor(self) -> None:
        trainer = object.__new__(PAFCTD3Trainer)
        trainer.torch = torch
        trainer.config = SimpleNamespace(
            economic_gt_grid_proxy_coef=0.25,
            exec_action_anchor_safe_floor=0.2,
            economic_teacher_bes_anchor_preserve_scale=0.85,
        )
        trainer.env_config = SimpleNamespace(
            q_boiler_cap_mw=10.0,
            abs_gate_scale_k=2.0,
        )
        trainer.observation_index = {}
        trainer.action_index = {"u_bes": 0}
        trainer._compute_gt_grid_proxy_terms = lambda **kwargs: {
            "price_advantage": torch.ones((1, 1), dtype=torch.float32),
            "net_grid_need_ratio": torch.ones((1, 1), dtype=torch.float32),
            "undercommit_ratio": torch.zeros((1, 1), dtype=torch.float32),
            "support_multiplier": torch.ones((1, 1), dtype=torch.float32),
        }
        trainer._compute_bes_price_prior_terms = lambda **kwargs: {
            "opportunity": torch.ones((1, 1), dtype=torch.float32),
            "target_u_bes": torch.ones((1, 1), dtype=torch.float32),
        }

        anchor = PAFCTD3Trainer._compute_gt_anchor_dimension_scale(
            trainer,
            obs_batch=torch.zeros((1, 1), dtype=torch.float32),
            action_raw_batch=torch.zeros((1, 1), dtype=torch.float32),
            action_exec_batch=torch.zeros((1, 1), dtype=torch.float32),
            teacher_available_batch=torch.ones((1, 1), dtype=torch.float32),
            teacher_action_mask_batch=torch.ones((1, 1), dtype=torch.float32),
        )

        self.assertAlmostEqual(float(anchor[0, 0].item()), 0.85, places=6)


if __name__ == "__main__":
    unittest.main()
