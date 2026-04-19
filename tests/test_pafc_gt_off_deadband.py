import unittest

from src.cchp_physical_env.policy.pafc_td3 import _canonicalize_gt_target_np


class PAFCGTOffDeadbandTest(unittest.TestCase):
    def test_tiny_gt_request_snaps_to_off(self) -> None:
        target = _canonicalize_gt_target_np(
            p_gt_target_mw=0.07,
            p_gt_low_mw=0.0,
            p_gt_high_mw=12.0,
            gt_min_output_mw=1.0,
            gt_off_deadband_ratio=0.5,
        )
        self.assertEqual(target, 0.0)

    def test_mid_submin_gt_request_still_snaps_to_min_output(self) -> None:
        target = _canonicalize_gt_target_np(
            p_gt_target_mw=0.7,
            p_gt_low_mw=0.0,
            p_gt_high_mw=12.0,
            gt_min_output_mw=1.0,
            gt_off_deadband_ratio=0.5,
        )
        self.assertEqual(target, 1.0)


if __name__ == "__main__":
    unittest.main()
