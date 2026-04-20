import unittest

from src.cchp_physical_env.core.config_loader import build_training_options


class ConfigLoaderPAFCTrustTest(unittest.TestCase):
    def test_build_training_options_accepts_surrogate_actor_trust_keys(self) -> None:
        options = build_training_options(
            {
                "policy": "pafc_td3",
                "pafc_projection_surrogate_checkpoint": "mock.pt",
                "pafc_surrogate_actor_trust_coef": 0.6,
                "pafc_surrogate_actor_trust_min_scale": 0.3,
            }
        )
        self.assertAlmostEqual(float(options["pafc_surrogate_actor_trust_coef"]), 0.6, places=8)
        self.assertAlmostEqual(
            float(options["pafc_surrogate_actor_trust_min_scale"]), 0.3, places=8
        )


if __name__ == "__main__":
    unittest.main()
