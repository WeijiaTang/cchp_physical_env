import unittest

from src.cchp_physical_env.policy.pafc_td3 import (
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


if __name__ == "__main__":
    unittest.main()
