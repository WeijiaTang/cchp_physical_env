import unittest

from src.cchp_physical_env.env.physics.tespy.gt_network import GTDesignPoint, GTNetwork


class GTNetworkMinOutputToleranceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.network = GTNetwork(GTDesignPoint(), build_tespy_topology=False)

    def test_near_min_output_keeps_gt_online(self) -> None:
        result = self.network.solve_offdesign(
            p_gt_request_mw=0.9999999999999998,
            t_amb_k=288.15,
        )
        self.assertAlmostEqual(result.p_gt_mw, 1.0, places=9)
        self.assertGreater(result.fuel_input_mw, 0.0)
        self.assertGreater(result.m_exh_kg_per_s, 0.0)

    def test_clear_sub_min_output_still_turns_gt_off(self) -> None:
        result = self.network.solve_offdesign(
            p_gt_request_mw=0.999,
            t_amb_k=288.15,
        )
        self.assertEqual(result.p_gt_mw, 0.0)
        self.assertEqual(result.fuel_input_mw, 0.0)
        self.assertEqual(result.m_exh_kg_per_s, 0.0)


if __name__ == "__main__":
    unittest.main()
