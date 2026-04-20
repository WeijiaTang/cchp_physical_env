import unittest
from types import SimpleNamespace

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


if __name__ == "__main__":
    unittest.main()
