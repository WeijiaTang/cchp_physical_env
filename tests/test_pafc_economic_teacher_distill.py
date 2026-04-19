import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.cchp_physical_env.policy.pafc_td3 import (
    PAFCTD3Trainer,
    PAFCTD3TrainConfig,
    _build_mixed_economic_teacher_target_np,
    _economic_teacher_blend_weight_np,
    _economic_teacher_gate_decision_np,
    _economic_teacher_projection_gap_np,
    _economic_teacher_target_mask_np,
    _estimate_dispatch_proxy_np,
    _materialize_teacher_action_np,
)


class PAFCEconomicTeacherDistillTest(unittest.TestCase):
    def test_prefill_replay_duplicates_teacher_target_transitions(self) -> None:
        class _ReplayStub:
            def __init__(self) -> None:
                self.entries: list[dict[str, object]] = []

            def add(self, **kwargs) -> None:
                self.entries.append(dict(kwargs))

        trainer = object.__new__(PAFCTD3Trainer)
        trainer.config = SimpleNamespace(
            expert_prefill_steps=2,
            expert_prefill_policy="checkpoint_dual",
            expert_prefill_checkpoint_path="safe_actor.json",
            expert_prefill_economic_policy="checkpoint",
            expert_prefill_economic_checkpoint_path="economic_actor.json",
            expert_prefill_cooling_bias=0.5,
            expert_prefill_abs_replay_boost=0,
            economic_teacher_prefill_replay_boost=2,
            expert_prefill_abs_exec_threshold=0.05,
            expert_prefill_abs_window_mining_candidates=1,
            episode_days=7,
            seed=42,
            dual_abs_margin_k=1.25,
            dual_qc_ratio_th=0.55,
            dual_heat_backup_ratio_th=0.10,
            dual_safe_abs_u_th=0.60,
        )
        trainer.observation_keys = ("price_e", "soc_bes")
        trainer.action_keys = ("u_gt", "u_bes", "u_boiler", "u_abs", "u_ech", "u_tes")
        trainer.action_index = {
            "u_gt": 0,
            "u_bes": 1,
            "u_boiler": 2,
            "u_abs": 3,
            "u_ech": 4,
            "u_tes": 5,
        }
        trainer.replay = _ReplayStub()
        trainer.economic_teacher_policy = object()
        trainer.economic_teacher_distill_summary = {}
        trainer._build_expert_policy = lambda: (object(), {"policy": "stub"})
        trainer._make_prefill_episode_df = lambda: [
            {"timestamp": "2025-01-01 00:00:00", "qc_dem_mw": 1.0},
            {"timestamp": "2025-01-01 00:15:00", "qc_dem_mw": 1.2},
        ]

        transitions = [
            {
                "obs": np.asarray([450.0, 0.20], dtype=np.float32),
                "next_obs": np.asarray([500.0, 0.22], dtype=np.float32),
                "action_raw": np.zeros((6,), dtype=np.float32),
                "action_exec": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
                "teacher_action_exec": np.asarray([0.0, -0.55, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
                "teacher_action_mask": np.asarray([0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
                "teacher_available": True,
                "reward": 0.0,
                "cost": np.zeros((3,), dtype=np.float32),
                "gap": np.zeros((3,), dtype=np.float32),
                "done": False,
                "teacher_source": "economic_checkpoint",
                "gate_reasons": [],
            },
            {
                "obs": np.asarray([1200.0, 0.80], dtype=np.float32),
                "next_obs": np.asarray([1100.0, 0.78], dtype=np.float32),
                "action_raw": np.zeros((6,), dtype=np.float32),
                "action_exec": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
                "teacher_action_exec": np.zeros((6,), dtype=np.float32),
                "teacher_action_mask": np.zeros((6,), dtype=np.float32),
                "teacher_available": False,
                "reward": 0.0,
                "cost": np.zeros((3,), dtype=np.float32),
                "gap": np.zeros((3,), dtype=np.float32),
                "done": True,
                "teacher_source": "",
                "gate_reasons": [],
            },
        ]
        trainer._rollout_expert_prefill_episode = lambda **kwargs: {
            "episode_df": trainer._make_prefill_episode_df(),
            "transitions": transitions,
            "mining_score": 1.0,
            "bes_prior_override_count": 0,
            "bes_prior_charge_count": 0,
            "bes_prior_discharge_count": 0,
        }

        outputs = PAFCTD3Trainer._prefill_replay_with_expert(trainer)

        self.assertEqual(len(outputs[0]), 4)
        self.assertEqual(len(trainer.replay.entries), 4)
        self.assertEqual(trainer.expert_prefill_summary["teacher_target_transition_count"], 1)
        self.assertEqual(trainer.expert_prefill_summary["teacher_replay_duplicates_added"], 2)
        self.assertEqual(trainer.economic_teacher_distill_summary["prefill_target_steps"], 1)

    def test_weight_increases_with_safety_and_disagreement(self) -> None:
        weak = _economic_teacher_blend_weight_np(
            opportunity_score=0.8,
            safety_margin=0.2,
            disagreement_score=0.2,
        )
        strong = _economic_teacher_blend_weight_np(
            opportunity_score=0.8,
            safety_margin=0.9,
            disagreement_score=0.9,
        )
        self.assertGreater(strong, weak)

    def test_zero_opportunity_zeroes_distill_weight(self) -> None:
        weight = _economic_teacher_blend_weight_np(
            opportunity_score=0.0,
            safety_margin=1.0,
            disagreement_score=1.0,
        )
        self.assertEqual(weight, 0.0)

    def test_materialize_teacher_action_bypasses_projection_for_executed_teacher(self) -> None:
        teacher_action = {
            "u_gt": 0.2,
            "u_bes": -0.1,
            "u_boiler": 0.3,
            "u_abs": 0.0,
            "u_ech": 0.4,
            "u_tes": 0.1,
        }

        def _unexpected_project(**kwargs):
            raise AssertionError(f"projector should not be called: {kwargs}")

        action_raw, action_exec = _materialize_teacher_action_np(
            teacher_action=teacher_action,
            action_keys=("u_gt", "u_bes", "u_boiler", "u_abs", "u_ech", "u_tes"),
            obs_vector=np.zeros((3,), dtype=np.float32),
            returns_executed_action=True,
            project_action_exec_fn=_unexpected_project,
        )
        np.testing.assert_allclose(action_raw, action_exec)

    def test_materialize_teacher_action_projects_checkpoint_teacher(self) -> None:
        teacher_action = {
            "u_gt": 0.2,
            "u_bes": -0.1,
            "u_boiler": 0.3,
            "u_abs": 0.0,
            "u_ech": 0.4,
            "u_tes": 0.1,
        }
        projected = np.asarray([0.1, -0.1, 0.25, 0.0, 0.2, 0.0], dtype=np.float32)

        def _project(**kwargs):
            return projected

        action_raw, action_exec = _materialize_teacher_action_np(
            teacher_action=teacher_action,
            action_keys=("u_gt", "u_bes", "u_boiler", "u_abs", "u_ech", "u_tes"),
            obs_vector=np.zeros((3,), dtype=np.float32),
            returns_executed_action=False,
            project_action_exec_fn=_project,
        )
        self.assertNotEqual(action_raw.tolist(), action_exec.tolist())
        np.testing.assert_allclose(action_exec, projected)

    def test_gate_rejects_low_advantage_against_safe_reference(self) -> None:
        decision = _economic_teacher_gate_decision_np(
            safe_reference_available=True,
            proxy_advantage_ratio=0.01,
            abs_risk_gap=0.0,
            projection_gap=0.05,
            min_proxy_advantage_ratio=0.02,
            max_safe_abs_risk_gap=0.05,
            max_projection_gap=0.20,
        )
        self.assertFalse(decision["accepted"])
        self.assertIn("proxy_advantage_low", decision["reasons"])

    def test_gate_rejects_large_projection_gap_without_safe_reference(self) -> None:
        decision = _economic_teacher_gate_decision_np(
            safe_reference_available=False,
            proxy_advantage_ratio=1.0,
            abs_risk_gap=0.0,
            projection_gap=0.35,
            min_proxy_advantage_ratio=0.02,
            max_safe_abs_risk_gap=0.05,
            max_projection_gap=0.20,
        )
        self.assertFalse(decision["accepted"])
        self.assertIn("projection_gap_high", decision["reasons"])

    def test_gate_accepts_stable_better_teacher(self) -> None:
        decision = _economic_teacher_gate_decision_np(
            safe_reference_available=True,
            proxy_advantage_ratio=0.08,
            abs_risk_gap=0.01,
            projection_gap=0.05,
            min_proxy_advantage_ratio=0.02,
            max_safe_abs_risk_gap=0.05,
            max_projection_gap=0.20,
        )
        self.assertTrue(decision["accepted"])
        self.assertEqual(decision["reasons"], [])

    def test_projection_gap_only_counts_distill_dimensions(self) -> None:
        action_index = {
            "u_gt": 0,
            "u_bes": 1,
            "u_boiler": 2,
            "u_abs": 3,
            "u_ech": 4,
            "u_tes": 5,
        }
        action_raw = np.asarray([0.2, 0.1, 0.0, 1.0, 1.0, -0.2], dtype=np.float32)
        action_exec = np.asarray([0.1, 0.0, 0.0, 0.0, 0.0, -0.1], dtype=np.float32)
        gap = _economic_teacher_projection_gap_np(
            action_raw=action_raw,
            action_exec=action_exec,
            action_index=action_index,
        )
        self.assertAlmostEqual(gap, 0.1, places=6)

    def test_projection_gap_can_be_restricted_to_teacher_mask(self) -> None:
        action_index = {
            "u_gt": 0,
            "u_bes": 1,
            "u_boiler": 2,
            "u_abs": 3,
            "u_ech": 4,
            "u_tes": 5,
        }
        action_raw = np.asarray([1.0, -0.2, 0.0, 0.0, 0.0, 0.9], dtype=np.float32)
        action_exec = np.asarray([0.0, -0.2, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        mask = np.asarray([0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        gap = _economic_teacher_projection_gap_np(
            action_raw=action_raw,
            action_exec=action_exec,
            action_index=action_index,
            teacher_mask=mask,
        )
        self.assertAlmostEqual(gap, 0.0, places=6)

    def test_teacher_target_mask_marks_selected_dimensions(self) -> None:
        action_index = {
            "u_gt": 0,
            "u_bes": 1,
            "u_boiler": 2,
            "u_abs": 3,
            "u_ech": 4,
            "u_tes": 5,
        }
        mask = _economic_teacher_target_mask_np(
            action_dim=6,
            action_index=action_index,
            keys=("u_gt", "u_tes"),
        )
        self.assertEqual(mask.tolist(), [1.0, 0.0, 0.0, 0.0, 0.0, 1.0])

    def test_mixed_teacher_swaps_only_profitable_stable_dimension(self) -> None:
        env_config = SimpleNamespace(
            dt_hours=0.25,
            p_gt_cap_mw=12.0,
            q_boiler_cap_mw=10.0,
            p_bes_cap_mw=4.0,
            q_ech_cap_mw=6.0,
            q_tes_charge_cap_mw=8.0,
            q_tes_discharge_cap_mw=8.0,
            cop_nominal=0.75,
            ech_cop_partload_min_fraction=0.72,
            gt_eta_min=0.26,
            gt_eta_max=0.36,
            gt_om_var_cost_per_mwh=20.0,
            gt_start_cost=250.0,
            gt_cycle_cost=250.0,
            bes_eta_charge=0.95,
            bes_eta_discharge=0.95,
            sell_price_ratio=0.20,
            sell_price_cap_per_mwh=300.0,
            abs_gate_scale_k=2.0,
            gt_min_output_mw=1.0,
        )
        action_index = {
            "u_gt": 0,
            "u_bes": 1,
            "u_boiler": 2,
            "u_abs": 3,
            "u_ech": 4,
            "u_tes": 5,
        }
        observation = {
            "p_dem_mw": 8.0,
            "pv_mw": 0.0,
            "wt_mw": 0.0,
            "price_e": 1200.0,
            "price_gas": 80.0,
            "t_amb_k": 298.15,
            "p_gt_prev_mw": 6.0,
            "abs_drive_margin_k": 5.0,
            "soc_bes": 0.6,
        }
        safe_action = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        economic_raw = np.asarray([1.0, 0.0, 0.0, 0.8, 0.8, 0.5], dtype=np.float32)
        economic_exec = np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.5], dtype=np.float32)
        safe_proxy = _estimate_dispatch_proxy_np(
            observation=observation,
            action_exec=safe_action,
            action_index=action_index,
            env_config=env_config,
            gt_off_deadband_ratio=0.0,
        )
        mixed = _build_mixed_economic_teacher_target_np(
            observation=observation,
            safe_action_exec=safe_action,
            safe_proxy=safe_proxy,
            economic_action_raw=economic_raw,
            economic_action_exec=economic_exec,
            action_index=action_index,
            env_config=env_config,
            gt_off_deadband_ratio=0.0,
            min_proxy_advantage_ratio=0.02,
            gt_proxy_advantage_ratio_min=0.01,
            max_safe_abs_risk_gap=0.05,
            max_projection_gap=0.20,
            gt_projection_gap_max=1.0,
            bes_proxy_advantage_ratio_min=0.002,
            bes_price_low_threshold=600.0,
            bes_price_high_threshold=1200.0,
            bes_charge_soc_ceiling=0.75,
            bes_discharge_soc_floor=0.35,
            bes_soc_min=0.10,
            bes_soc_max=0.95,
            bes_charge_u=0.63,
            bes_discharge_u=0.35,
            bes_price_opportunity_min=0.10,
            gt_abs_margin_guard_k=1.25,
            gt_qc_ratio_guard=0.55,
            gt_heat_backup_ratio_guard=0.10,
        )
        self.assertEqual(mixed["swapped_dims"], ["u_gt"])
        self.assertEqual(mixed["mask"].tolist(), [1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def test_mixed_teacher_allows_gt_with_relaxed_gt_projection_gap(self) -> None:
        env_config = SimpleNamespace(
            dt_hours=0.25,
            p_gt_cap_mw=12.0,
            q_boiler_cap_mw=10.0,
            p_bes_cap_mw=4.0,
            q_abs_cool_cap_mw=4.5,
            q_ech_cap_mw=6.0,
            q_tes_charge_cap_mw=8.0,
            q_tes_discharge_cap_mw=8.0,
            cop_nominal=0.75,
            ech_cop_partload_min_fraction=0.72,
            gt_eta_min=0.26,
            gt_eta_max=0.36,
            gt_om_var_cost_per_mwh=20.0,
            gt_start_cost=250.0,
            gt_cycle_cost=250.0,
            bes_eta_charge=0.95,
            bes_eta_discharge=0.95,
            sell_price_ratio=0.20,
            sell_price_cap_per_mwh=300.0,
            abs_gate_scale_k=2.0,
            gt_min_output_mw=1.0,
        )
        action_index = {
            "u_gt": 0,
            "u_bes": 1,
            "u_boiler": 2,
            "u_abs": 3,
            "u_ech": 4,
            "u_tes": 5,
        }
        observation = {
            "p_dem_mw": 9.0,
            "pv_mw": 0.0,
            "wt_mw": 0.0,
            "price_e": 1300.0,
            "price_gas": 80.0,
            "t_amb_k": 298.15,
            "p_gt_prev_mw": 5.0,
            "abs_drive_margin_k": 4.0,
            "soc_bes": 0.55,
        }
        safe_action = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        economic_raw = np.asarray([0.0, 0.0, 0.0, 0.9, 0.9, 0.0], dtype=np.float32)
        economic_exec = np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        safe_proxy = _estimate_dispatch_proxy_np(
            observation=observation,
            action_exec=safe_action,
            action_index=action_index,
            env_config=env_config,
            gt_off_deadband_ratio=0.0,
        )
        mixed = _build_mixed_economic_teacher_target_np(
            observation=observation,
            safe_action_exec=safe_action,
            safe_proxy=safe_proxy,
            economic_action_raw=economic_raw,
            economic_action_exec=economic_exec,
            action_index=action_index,
            env_config=env_config,
            gt_off_deadband_ratio=0.0,
            min_proxy_advantage_ratio=0.02,
            gt_proxy_advantage_ratio_min=0.01,
            max_safe_abs_risk_gap=0.05,
            max_projection_gap=0.20,
            gt_projection_gap_max=1.10,
            bes_proxy_advantage_ratio_min=0.002,
            bes_price_low_threshold=600.0,
            bes_price_high_threshold=1200.0,
            bes_charge_soc_ceiling=0.75,
            bes_discharge_soc_floor=0.35,
            bes_soc_min=0.10,
            bes_soc_max=0.95,
            bes_charge_u=0.63,
            bes_discharge_u=0.35,
            bes_price_opportunity_min=0.10,
            gt_abs_margin_guard_k=1.25,
            gt_qc_ratio_guard=0.55,
            gt_heat_backup_ratio_guard=0.10,
        )
        self.assertEqual(mixed["swapped_dims"], ["u_gt"])
        self.assertEqual(mixed["mask"].tolist(), [1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def test_mixed_teacher_blocks_gt_swap_when_cooling_is_critical(self) -> None:
        env_config = SimpleNamespace(
            dt_hours=0.25,
            p_gt_cap_mw=12.0,
            q_boiler_cap_mw=10.0,
            p_bes_cap_mw=4.0,
            q_abs_cool_cap_mw=4.5,
            q_ech_cap_mw=6.0,
            q_tes_charge_cap_mw=8.0,
            q_tes_discharge_cap_mw=8.0,
            cop_nominal=0.75,
            ech_cop_partload_min_fraction=0.72,
            gt_eta_min=0.26,
            gt_eta_max=0.36,
            gt_om_var_cost_per_mwh=20.0,
            gt_start_cost=250.0,
            gt_cycle_cost=250.0,
            bes_eta_charge=0.95,
            bes_eta_discharge=0.95,
            sell_price_ratio=0.20,
            sell_price_cap_per_mwh=300.0,
            abs_gate_scale_k=2.0,
            gt_min_output_mw=1.0,
        )
        action_index = {
            "u_gt": 0,
            "u_bes": 1,
            "u_boiler": 2,
            "u_abs": 3,
            "u_ech": 4,
            "u_tes": 5,
        }
        observation = {
            "p_dem_mw": 8.0,
            "pv_mw": 0.0,
            "wt_mw": 0.0,
            "price_e": 1200.0,
            "price_gas": 80.0,
            "t_amb_k": 298.15,
            "p_gt_prev_mw": 6.0,
            "abs_drive_margin_k": 0.5,
            "qc_dem_mw": 6.2,
            "heat_backup_min_needed_mw": 1.5,
            "soc_bes": 0.7,
        }
        safe_action = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        economic_raw = np.asarray([1.0, 0.8, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        economic_exec = np.asarray([1.0, 0.8, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        safe_proxy = _estimate_dispatch_proxy_np(
            observation=observation,
            action_exec=safe_action,
            action_index=action_index,
            env_config=env_config,
            gt_off_deadband_ratio=0.0,
        )
        mixed = _build_mixed_economic_teacher_target_np(
            observation=observation,
            safe_action_exec=safe_action,
            safe_proxy=safe_proxy,
            economic_action_raw=economic_raw,
            economic_action_exec=economic_exec,
            action_index=action_index,
            env_config=env_config,
            gt_off_deadband_ratio=0.0,
            min_proxy_advantage_ratio=0.02,
            gt_proxy_advantage_ratio_min=0.01,
            max_safe_abs_risk_gap=0.05,
            max_projection_gap=0.20,
            gt_projection_gap_max=1.0,
            bes_proxy_advantage_ratio_min=0.002,
            bes_price_low_threshold=600.0,
            bes_price_high_threshold=1200.0,
            bes_charge_soc_ceiling=0.75,
            bes_discharge_soc_floor=0.35,
            bes_soc_min=0.10,
            bes_soc_max=0.95,
            bes_charge_u=0.63,
            bes_discharge_u=0.35,
            bes_price_opportunity_min=0.10,
            gt_abs_margin_guard_k=1.25,
            gt_qc_ratio_guard=0.55,
            gt_heat_backup_ratio_guard=0.10,
        )
        self.assertNotIn("u_gt", mixed["swapped_dims"])
        self.assertIn("u_bes", mixed["swapped_dims"])

    def test_mixed_teacher_accepts_bes_with_small_but_positive_advantage_when_price_signal_matches(self) -> None:
        env_config = SimpleNamespace(
            dt_hours=0.25,
            p_gt_cap_mw=12.0,
            q_boiler_cap_mw=10.0,
            p_bes_cap_mw=4.0,
            q_abs_cool_cap_mw=4.5,
            q_ech_cap_mw=6.0,
            q_tes_charge_cap_mw=8.0,
            q_tes_discharge_cap_mw=8.0,
            cop_nominal=0.75,
            ech_cop_partload_min_fraction=0.72,
            gt_eta_min=0.26,
            gt_eta_max=0.36,
            gt_om_var_cost_per_mwh=20.0,
            gt_start_cost=250.0,
            gt_cycle_cost=250.0,
            bes_eta_charge=0.95,
            bes_eta_discharge=0.95,
            sell_price_ratio=0.20,
            sell_price_cap_per_mwh=300.0,
            abs_gate_scale_k=2.0,
            gt_min_output_mw=1.0,
        )
        action_index = {
            "u_gt": 0,
            "u_bes": 1,
            "u_boiler": 2,
            "u_abs": 3,
            "u_ech": 4,
            "u_tes": 5,
        }
        observation = {
            "p_dem_mw": 4.0,
            "pv_mw": 0.0,
            "wt_mw": 0.0,
            "price_e": 500.0,
            "price_gas": 80.0,
            "t_amb_k": 298.15,
            "p_gt_prev_mw": 0.0,
            "abs_drive_margin_k": 5.0,
            "soc_bes": 0.25,
            "qc_dem_mw": 1.0,
            "heat_backup_min_needed_mw": 0.0,
        }
        safe_action = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        economic_raw = np.asarray([0.0, -0.2, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        economic_exec = np.asarray([0.0, -0.2, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        safe_proxy = _estimate_dispatch_proxy_np(
            observation=observation,
            action_exec=safe_action,
            action_index=action_index,
            env_config=env_config,
            gt_off_deadband_ratio=0.0,
        )
        mixed = _build_mixed_economic_teacher_target_np(
            observation=observation,
            safe_action_exec=safe_action,
            safe_proxy=safe_proxy,
            economic_action_raw=economic_raw,
            economic_action_exec=economic_exec,
            action_index=action_index,
            env_config=env_config,
            gt_off_deadband_ratio=0.0,
            min_proxy_advantage_ratio=0.02,
            gt_proxy_advantage_ratio_min=0.01,
            max_safe_abs_risk_gap=0.05,
            max_projection_gap=0.20,
            gt_projection_gap_max=1.0,
            bes_proxy_advantage_ratio_min=0.002,
            bes_price_low_threshold=600.0,
            bes_price_high_threshold=1200.0,
            bes_charge_soc_ceiling=0.75,
            bes_discharge_soc_floor=0.35,
            bes_soc_min=0.10,
            bes_soc_max=0.95,
            bes_charge_u=0.63,
            bes_discharge_u=0.35,
            bes_price_opportunity_min=0.10,
            gt_abs_margin_guard_k=1.25,
            gt_qc_ratio_guard=0.55,
            gt_heat_backup_ratio_guard=0.10,
        )
        self.assertIn("u_bes", mixed["swapped_dims"])

    def test_mixed_teacher_blocks_bes_when_sign_conflicts_with_price_signal(self) -> None:
        env_config = SimpleNamespace(
            dt_hours=0.25,
            p_gt_cap_mw=12.0,
            q_boiler_cap_mw=10.0,
            p_bes_cap_mw=4.0,
            q_abs_cool_cap_mw=4.5,
            q_ech_cap_mw=6.0,
            q_tes_charge_cap_mw=8.0,
            q_tes_discharge_cap_mw=8.0,
            cop_nominal=0.75,
            ech_cop_partload_min_fraction=0.72,
            gt_eta_min=0.26,
            gt_eta_max=0.36,
            gt_om_var_cost_per_mwh=20.0,
            gt_start_cost=250.0,
            gt_cycle_cost=250.0,
            bes_eta_charge=0.95,
            bes_eta_discharge=0.95,
            sell_price_ratio=0.20,
            sell_price_cap_per_mwh=300.0,
            abs_gate_scale_k=2.0,
            gt_min_output_mw=1.0,
        )
        action_index = {
            "u_gt": 0,
            "u_bes": 1,
            "u_boiler": 2,
            "u_abs": 3,
            "u_ech": 4,
            "u_tes": 5,
        }
        observation = {
            "p_dem_mw": 4.0,
            "pv_mw": 0.0,
            "wt_mw": 0.0,
            "price_e": 500.0,
            "price_gas": 80.0,
            "t_amb_k": 298.15,
            "p_gt_prev_mw": 0.0,
            "abs_drive_margin_k": 5.0,
            "soc_bes": 0.25,
            "qc_dem_mw": 1.0,
            "heat_backup_min_needed_mw": 0.0,
        }
        safe_action = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        economic_raw = np.asarray([0.0, 0.2, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        economic_exec = np.asarray([0.0, 0.2, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        safe_proxy = _estimate_dispatch_proxy_np(
            observation=observation,
            action_exec=safe_action,
            action_index=action_index,
            env_config=env_config,
            gt_off_deadband_ratio=0.0,
        )
        mixed = _build_mixed_economic_teacher_target_np(
            observation=observation,
            safe_action_exec=safe_action,
            safe_proxy=safe_proxy,
            economic_action_raw=economic_raw,
            economic_action_exec=economic_exec,
            action_index=action_index,
            env_config=env_config,
            gt_off_deadband_ratio=0.0,
            min_proxy_advantage_ratio=0.02,
            gt_proxy_advantage_ratio_min=0.01,
            max_safe_abs_risk_gap=0.05,
            max_projection_gap=0.20,
            gt_projection_gap_max=1.0,
            bes_proxy_advantage_ratio_min=0.002,
            bes_price_low_threshold=600.0,
            bes_price_high_threshold=1200.0,
            bes_charge_soc_ceiling=0.75,
            bes_discharge_soc_floor=0.35,
            bes_soc_min=0.10,
            bes_soc_max=0.95,
            bes_charge_u=0.63,
            bes_discharge_u=0.35,
            bes_price_opportunity_min=0.10,
            gt_abs_margin_guard_k=1.25,
            gt_qc_ratio_guard=0.55,
            gt_heat_backup_ratio_guard=0.10,
        )
        self.assertNotIn("u_bes", mixed["swapped_dims"])

    def test_dispatch_proxy_penalizes_unnecessary_gt_start(self) -> None:
        env_config = SimpleNamespace(
            dt_hours=0.25,
            p_gt_cap_mw=12.0,
            q_boiler_cap_mw=10.0,
            p_bes_cap_mw=4.0,
            q_ech_cap_mw=6.0,
            cop_nominal=0.75,
            ech_cop_partload_min_fraction=0.72,
            gt_eta_min=0.26,
            gt_eta_max=0.36,
            gt_om_var_cost_per_mwh=20.0,
            gt_start_cost=250.0,
            gt_cycle_cost=250.0,
            bes_eta_charge=0.95,
            bes_eta_discharge=0.95,
            sell_price_ratio=0.20,
            sell_price_cap_per_mwh=300.0,
            abs_gate_scale_k=2.0,
            gt_min_output_mw=1.0,
        )
        action_index = {
            "u_gt": 0,
            "u_bes": 1,
            "u_boiler": 2,
            "u_abs": 3,
            "u_ech": 4,
            "u_tes": 5,
        }
        observation = {
            "p_dem_mw": 1.0,
            "pv_mw": 0.0,
            "wt_mw": 0.0,
            "price_e": 300.0,
            "price_gas": 400.0,
            "t_amb_k": 298.15,
            "p_gt_prev_mw": 0.0,
            "abs_drive_margin_k": 5.0,
        }
        gt_off = np.asarray([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        gt_on = np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        off_cost = _estimate_dispatch_proxy_np(
            observation=observation,
            action_exec=gt_off,
            action_index=action_index,
            env_config=env_config,
            gt_off_deadband_ratio=0.0,
        )["proxy_cost"]
        on_cost = _estimate_dispatch_proxy_np(
            observation=observation,
            action_exec=gt_on,
            action_index=action_index,
            env_config=env_config,
            gt_off_deadband_ratio=0.0,
        )["proxy_cost"]
        self.assertGreater(on_cost, off_cost)

    def test_planner_economic_teacher_policy_does_not_require_checkpoint_path(self) -> None:
        config = PAFCTD3TrainConfig(
            projection_surrogate_checkpoint_path=Path(__file__),
            expert_prefill_policy="checkpoint_dual",
            expert_prefill_checkpoint_path=Path(__file__),
            expert_prefill_economic_policy="milp_mpc",
            expert_prefill_economic_checkpoint_path="",
            economic_teacher_distill_coef=1.0,
        )
        self.assertEqual(config.expert_prefill_economic_policy, "milp_mpc")


if __name__ == "__main__":
    unittest.main()
