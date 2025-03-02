from brief import BriefInformation

import unittest


class TestBriefInformation(unittest.TestCase):

    def test_initial_timeline_okay_2_variants(
        self,
    ):
        RELATIVE_DELTA_THRESHOLD = 0.12
        ALPHA, POWER = 0.05, 0.8
        NUM_VARIANTS = 2
        MONTHLY_REACH, BASELINE, EXPERIMENT_SIZING_TIMELINE = 206800, 0.6014, 3
        brief = BriefInformation(
            monthly_reach=MONTHLY_REACH,
            baseline=BASELINE,
            experiment_sizing_timeline=EXPERIMENT_SIZING_TIMELINE,
            num_variants=NUM_VARIANTS,
            alpha=ALPHA,
            power=POWER,
            relative_threshold=RELATIVE_DELTA_THRESHOLD,
        )
        (
            initial_bucket_size,
            initial_absolute_delta,
            initial_relative_delta,
            initial_timeline_in_months,
            initial_timeline_in_weeks,
        ) = brief.compute_brief_information()
        self.assertEqual(initial_bucket_size, 77550)
        self.assertAlmostEqual(initial_absolute_delta, 0.00617, places=3)
        self.assertAlmostEqual(initial_relative_delta, 0.0102, places=3)
        self.assertEqual(initial_timeline_in_months, 0.75)
        self.assertEqual(initial_timeline_in_weeks, 3.0)

    def test_initial_timeline_okay():
        RELATIVE_DELTA_THRESHOLD = 0.12
        ALPHA, POWER = 0.05, 0.8
        NUM_VARIANTS = 2
        MONTHLY_REACH, BASELINE, EXPERIMENT_SIZING_TIMELINE = 206800, 0.6014, 3
        brief = BriefInformation(
            monthly_reach=MONTHLY_REACH,
            baseline=BASELINE,
            experiment_sizing_timeline=EXPERIMENT_SIZING_TIMELINE,
            num_variants=NUM_VARIANTS,
            alpha=ALPHA,
            power=POWER,
            relative_threshold=RELATIVE_DELTA_THRESHOLD,
        )
        brief.compute_brief_information()

    def test_initial_timeline_long():
        RELATIVE_DELTA_THRESHOLD = 0.12
        ALPHA, POWER = 0.05, 0.8
        NUM_VARIANTS = 2
        MONTHLY_REACH, BASELINE, EXPERIMENT_SIZING_TIMELINE = 6883, 0.0486, 3
        brief = BriefInformation(
            monthly_reach=MONTHLY_REACH,
            baseline=BASELINE,
            experiment_sizing_timeline=EXPERIMENT_SIZING_TIMELINE,
            num_variants=NUM_VARIANTS,
            alpha=ALPHA,
            power=POWER,
            relative_threshold=RELATIVE_DELTA_THRESHOLD,
        )
        brief.compute_brief_information()

    def test_timeline_3_variants_faster():
        RELATIVE_DELTA_THRESHOLD = 0.12
        ALPHA, POWER = 0.05, 0.8
        NUM_VARIANTS = 3
        MONTHLY_REACH, BASELINE, EXPERIMENT_SIZING_TIMELINE = 50000, 0.4171, 3
        brief = BriefInformation(
            monthly_reach=MONTHLY_REACH,
            baseline=BASELINE,
            experiment_sizing_timeline=EXPERIMENT_SIZING_TIMELINE,
            num_variants=NUM_VARIANTS,
            alpha=ALPHA,
            power=POWER,
            relative_threshold=RELATIVE_DELTA_THRESHOLD,
        )
        brief.compute_brief_information()

    def test_initial_timeline_okay_3_variants(
        self,
    ):
        pass

    def test_initial_timeline_long_2_variants(
        self,
    ):
        pass

    def test_initial_timeline_long_3_variants(
        self,
    ):
        pass


if __name__ == "__main__":
    unittest.main()
