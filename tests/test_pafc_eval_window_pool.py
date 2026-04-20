import unittest
from types import SimpleNamespace

import pandas as pd

from src.cchp_physical_env.policy.pafc_td3 import (
    _allocate_economic_teacher_action_counts,
    _build_eval_window_pool,
    _select_temporal_priority_indices_by_season,
)


def _make_eval_window_df() -> pd.DataFrame:
    window_specs = (
        ("2024-01-01", 0.05, 2.5),
        ("2024-01-08", 0.10, 2.4),
        ("2024-04-01", 0.30, 1.8),
        ("2024-04-08", 0.35, 1.7),
        ("2024-07-01", 3.20, 0.4),
        ("2024-07-08", 3.60, 0.3),
        ("2024-10-01", 0.45, 1.2),
        ("2024-10-08", 0.40, 1.1),
    )
    frames: list[pd.DataFrame] = []
    for start_date, qc_level, qh_level in window_specs:
        timestamp = pd.date_range(
            start=start_date,
            periods=7 * 96,
            freq="15min",
            tz="Asia/Shanghai",
        )
        frames.append(
            pd.DataFrame(
                {
                    "timestamp": timestamp,
                    "qc_dem_mw": [qc_level] * len(timestamp),
                    "qh_dem_mw": [qh_level] * len(timestamp),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


class PAFCEvalWindowPoolTest(unittest.TestCase):
    def setUp(self) -> None:
        self.train_df = _make_eval_window_df()
        self.env_config = SimpleNamespace(dt_hours=0.25)

    def test_single_window_selection_prefers_cooling_critical_window(self) -> None:
        result = _build_eval_window_pool(
            train_df=self.train_df,
            env_config=self.env_config,
            eval_episode_days=7,
            pool_size=1,
            window_count=1,
            seed=42,
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["mode"], "season_balanced_multi_window_pool_v2")
        self.assertEqual(result["selected_season_counts"]["summer"], 1)
        selected = [
            item
            for item in result["windows"]
            if item["selected_for_eval"]
        ]
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["start_timestamp"], "2024-07-08T00:00:00+08:00")
        self.assertEqual(selected[0]["season"], "summer")

    def test_four_window_selection_covers_all_seasons(self) -> None:
        result = _build_eval_window_pool(
            train_df=self.train_df,
            env_config=self.env_config,
            eval_episode_days=7,
            pool_size=4,
            window_count=4,
            seed=42,
        )
        self.assertIsNotNone(result)
        selected = [
            item
            for item in result["windows"]
            if item["selected_for_eval"]
        ]
        self.assertEqual(len(selected), 4)
        self.assertEqual(
            {item["season"] for item in selected},
            {"winter", "spring", "summer", "autumn"},
        )
        self.assertEqual(result["selected_season_counts"]["summer"], 1)


class PAFCEconomicTeacherSelectionTest(unittest.TestCase):
    def test_allocate_economic_teacher_action_counts_reserves_bes_budget(self) -> None:
        counts = _allocate_economic_teacher_action_counts(
            requested_total=64,
            gt_available=13536,
            bes_available=2898,
            tes_available=0,
        )
        self.assertEqual(counts["gt"] + counts["bes"] + counts["tes"], 64)
        self.assertGreaterEqual(counts["bes"], 16)
        self.assertGreater(counts["gt"], counts["bes"])

    def test_select_temporal_priority_indices_by_season_keeps_summer_windows(self) -> None:
        priorities = [0.2, 0.3, 0.5, 0.6, 2.0, 2.2, 0.7, 0.8]
        season_by_index = {
            0: "winter",
            1: "winter",
            2: "spring",
            3: "spring",
            4: "summer",
            5: "summer",
            6: "autumn",
            7: "autumn",
        }
        selected = _select_temporal_priority_indices_by_season(
            indices=list(range(8)),
            priorities=priorities,
            season_by_index=season_by_index,
            target_count=4,
        )
        self.assertEqual(len(selected), 4)
        self.assertIn(5, selected)
        self.assertEqual(
            {season_by_index[idx] for idx in selected},
            {"winter", "spring", "summer", "autumn"},
        )


if __name__ == "__main__":
    unittest.main()
