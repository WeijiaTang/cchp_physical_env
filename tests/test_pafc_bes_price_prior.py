import unittest

from src.cchp_physical_env.policy.pafc_td3 import _bes_price_prior_target_np


class PAFCBESPricePriorTest(unittest.TestCase):
    def test_low_price_with_soc_headroom_prefers_charging(self) -> None:
        prior = _bes_price_prior_target_np(
            price_e=250.0,
            soc_bes=0.25,
            price_low_threshold=300.0,
            price_high_threshold=1500.0,
            charge_soc_ceiling=0.75,
            discharge_soc_floor=0.35,
            bes_soc_min=0.10,
            bes_soc_max=0.95,
            charge_u=0.35,
            discharge_u=0.35,
        )
        self.assertEqual(prior["mode"], "charge")
        self.assertLess(float(prior["target_u_bes"]), 0.0)
        self.assertGreater(float(prior["opportunity"]), 0.0)

    def test_high_price_with_energy_prefers_discharging(self) -> None:
        prior = _bes_price_prior_target_np(
            price_e=1600.0,
            soc_bes=0.70,
            price_low_threshold=300.0,
            price_high_threshold=1500.0,
            charge_soc_ceiling=0.75,
            discharge_soc_floor=0.35,
            bes_soc_min=0.10,
            bes_soc_max=0.95,
            charge_u=0.35,
            discharge_u=0.35,
        )
        self.assertEqual(prior["mode"], "discharge")
        self.assertGreater(float(prior["target_u_bes"]), 0.0)
        self.assertGreater(float(prior["opportunity"]), 0.0)

    def test_mid_price_or_low_soc_keeps_bes_idle(self) -> None:
        mid_price = _bes_price_prior_target_np(
            price_e=900.0,
            soc_bes=0.50,
            price_low_threshold=300.0,
            price_high_threshold=1500.0,
            charge_soc_ceiling=0.75,
            discharge_soc_floor=0.35,
            bes_soc_min=0.10,
            bes_soc_max=0.95,
            charge_u=0.35,
            discharge_u=0.35,
        )
        depleted_high_price = _bes_price_prior_target_np(
            price_e=1600.0,
            soc_bes=0.20,
            price_low_threshold=300.0,
            price_high_threshold=1500.0,
            charge_soc_ceiling=0.75,
            discharge_soc_floor=0.35,
            bes_soc_min=0.10,
            bes_soc_max=0.95,
            charge_u=0.35,
            discharge_u=0.35,
        )
        self.assertEqual(mid_price["mode"], "idle")
        self.assertEqual(float(mid_price["target_u_bes"]), 0.0)
        self.assertEqual(depleted_high_price["mode"], "idle")
        self.assertEqual(float(depleted_high_price["target_u_bes"]), 0.0)

    def test_shoulder_price_emits_weak_charge_signal_when_soc_has_headroom(self) -> None:
        prior = _bes_price_prior_target_np(
            price_e=850.0,
            soc_bes=0.30,
            price_low_threshold=300.0,
            price_high_threshold=1500.0,
            charge_soc_ceiling=0.75,
            discharge_soc_floor=0.35,
            bes_soc_min=0.10,
            bes_soc_max=0.95,
            charge_u=0.35,
            discharge_u=0.35,
        )
        self.assertEqual(prior["mode"], "charge")
        self.assertLess(float(prior["target_u_bes"]), 0.0)
        self.assertLess(abs(float(prior["target_u_bes"])), 0.35)
        self.assertGreater(float(prior["opportunity"]), 0.05)


if __name__ == "__main__":
    unittest.main()
