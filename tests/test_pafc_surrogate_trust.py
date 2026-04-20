import unittest

import numpy as np

from src.cchp_physical_env.policy.pafc_td3 import (
    _blend_surrogate_action_proxy,
    _build_surrogate_actor_trust_scale_np,
)


class PAFCSurrogateTrustTest(unittest.TestCase):
    def test_surrogate_actor_trust_penalizes_high_error_dimensions(self) -> None:
        scales, summary = _build_surrogate_actor_trust_scale_np(
            action_keys=("u_gt", "u_bes", "u_boiler"),
            overall_mae_by_action={"u_gt": 0.10, "u_bes": 0.60, "u_boiler": 1.20},
            focused_mae_by_action={"u_gt": 0.10, "u_bes": 0.80, "u_boiler": 1.60},
            trust_coef=0.75,
            trust_min_scale=0.25,
        )
        self.assertEqual(scales.shape, (1, 3))
        self.assertTrue(bool(summary["enabled"]))
        self.assertAlmostEqual(float(scales[0, 0]), 0.925, places=5)
        self.assertLess(float(scales[0, 1]), float(scales[0, 0]))
        self.assertAlmostEqual(float(scales[0, 2]), 0.25, places=6)
        self.assertEqual(str(summary["weakest_action_by_trust"]), "u_boiler")

    def test_surrogate_actor_trust_disabled_returns_identity(self) -> None:
        scales, summary = _build_surrogate_actor_trust_scale_np(
            action_keys=("u_gt", "u_bes"),
            overall_mae_by_action={"u_gt": 0.4, "u_bes": 0.9},
            focused_mae_by_action={"u_gt": 0.5, "u_bes": 1.1},
            trust_coef=0.0,
            trust_min_scale=0.25,
        )
        np.testing.assert_allclose(scales, np.ones((1, 2), dtype=np.float32))
        self.assertFalse(bool(summary["enabled"]))
        self.assertEqual(str(summary["status"]), "disabled")

    def test_blend_surrogate_action_proxy_falls_back_on_low_trust_dims(self) -> None:
        blended = _blend_surrogate_action_proxy(
            action_exec_hat=np.asarray([[1.0, 0.2, -0.5]], dtype=np.float32),
            fallback_action=np.asarray([[0.0, 0.8, 0.5]], dtype=np.float32),
            trust_weight=np.asarray([[0.25, 1.0, 0.0]], dtype=np.float32),
        )
        np.testing.assert_allclose(
            blended,
            np.asarray([[0.25, 0.2, 0.5]], dtype=np.float32),
        )


if __name__ == "__main__":
    unittest.main()
