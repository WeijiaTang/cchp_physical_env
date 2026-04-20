import unittest

from src.cchp_physical_env.__main__ import build_parser


class TestPafcCliArgs(unittest.TestCase):
    def test_train_accepts_safe_preserve_overrides(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "train",
                "--policy",
                "pafc_td3",
                "--pafc-economic-teacher-safe-preserve-coef",
                "3.0",
                "--pafc-economic-teacher-safe-preserve-low-margin-boost",
                "1.0",
                "--pafc-economic-teacher-safe-preserve-high-cooling-boost",
                "1.5",
                "--pafc-economic-teacher-safe-preserve-joint-boost",
                "2.0",
            ]
        )
        self.assertEqual(args.command, "train")
        self.assertAlmostEqual(args.pafc_economic_teacher_safe_preserve_coef, 3.0)
        self.assertAlmostEqual(args.pafc_economic_teacher_safe_preserve_low_margin_boost, 1.0)
        self.assertAlmostEqual(args.pafc_economic_teacher_safe_preserve_high_cooling_boost, 1.5)
        self.assertAlmostEqual(args.pafc_economic_teacher_safe_preserve_joint_boost, 2.0)

    def test_pafc_train_accepts_gt_teacher_gate_overrides(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "pafc-train",
                "--projection-surrogate-checkpoint",
                "dummy.pt",
                "--economic-teacher-gt-proxy-advantage-min",
                "0.01",
                "--economic-teacher-gt-projection-gap-max",
                "1.25",
            ]
        )
        self.assertEqual(args.command, "pafc-train")
        self.assertAlmostEqual(args.economic_teacher_gt_proxy_advantage_min, 0.01)
        self.assertAlmostEqual(args.economic_teacher_gt_projection_gap_max, 1.25)


if __name__ == "__main__":
    unittest.main()
