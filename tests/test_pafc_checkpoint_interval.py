import unittest
from types import SimpleNamespace

from src.cchp_physical_env.policy import pafc_td3
from src.cchp_physical_env.policy.pafc_td3 import PAFCTD3Trainer


class PAFCCheckpointIntervalResolutionTest(unittest.TestCase):
    def _build_trainer(self, *, checkpoint_interval_steps: int, total_env_steps: int) -> PAFCTD3Trainer:
        trainer = object.__new__(PAFCTD3Trainer)
        trainer.config = SimpleNamespace(
            checkpoint_interval_steps=int(checkpoint_interval_steps),
            total_env_steps=int(total_env_steps),
        )
        return trainer

    def test_explicit_checkpoint_interval_is_respected(self) -> None:
        trainer = self._build_trainer(checkpoint_interval_steps=48, total_env_steps=256)
        self.assertEqual(trainer._resolve_checkpoint_interval_steps(), 48)

    def test_auto_checkpoint_interval_uses_dense_cadence_for_128_steps(self) -> None:
        trainer = self._build_trainer(checkpoint_interval_steps=0, total_env_steps=128)
        self.assertEqual(trainer._resolve_checkpoint_interval_steps(), 32)

    def test_auto_checkpoint_interval_uses_dense_cadence_for_256_steps(self) -> None:
        trainer = self._build_trainer(checkpoint_interval_steps=0, total_env_steps=256)
        self.assertEqual(trainer._resolve_checkpoint_interval_steps(), 64)

    def test_auto_checkpoint_interval_preserves_large_budget_behavior(self) -> None:
        trainer = self._build_trainer(checkpoint_interval_steps=0, total_env_steps=4096)
        self.assertEqual(trainer._resolve_checkpoint_interval_steps(), 1024)


class PAFCPostTrainRerankCandidateTest(unittest.TestCase):
    def test_candidates_keep_reward_leader_and_last_for_posttrain_rerank(self) -> None:
        selector = getattr(pafc_td3, "_select_posttrain_rerank_candidates", None)
        self.assertIsNotNone(selector)
        if selector is None:
            return

        history_items = [
            {
                "checkpoint_path": "selected.pt",
                "timesteps": 16_384,
                "mean_total_cost": 1_000.0,
                "reliability_min": {"electric": 1.0, "heat": 1.0, "cooling": 1.0},
                "gate": {"passed": True, "shortfall": {"total": 0.0, "max": 0.0}},
            },
            {
                "checkpoint_path": "reward.pt",
                "timesteps": 98_304,
                "mean_total_cost": 500.0,
                "reliability_min": {"electric": 1.0, "heat": 1.0, "cooling": 0.96},
                "gate": {"passed": False, "shortfall": {"total": 0.03, "max": 0.03}},
            },
        ]
        candidates = selector(
            history_items=history_items,
            selected_snapshot={"checkpoint_path": "selected.pt", "timesteps": 16_384},
            reward_snapshot={"checkpoint_path": "reward.pt", "timesteps": 98_304},
            last_snapshot={"checkpoint_path": "last.pt", "timesteps": 212_992},
            top_k=3,
            path_exists=lambda _: True,
        )

        self.assertEqual(
            [candidate["checkpoint_path"] for candidate in candidates],
            ["selected.pt", "reward.pt", "last.pt"],
        )


if __name__ == "__main__":
    unittest.main()
