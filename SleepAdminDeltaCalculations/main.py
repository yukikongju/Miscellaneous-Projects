import pandas as pd

from utils import calculateMetricParameters


class BriefInformation:

    def __init__(self, monthly_reach: int, baseline: float, experiment_sizing_timeline: int, num_variants: int, power=0.8, alpha=0.05, relative_threshold=0.12):
        """
        Arguments
        ---------
        monthly_reach: int
            number of user reached. This information is found in MixPanel
        baseline: float
            conversion rate between 0 and 1. This information is found in MixPanel
        experiment_sizing_timeline: int
            week neededed to complete this experiment in weeks. This information
            can be found in the 'Experiment Sizing Tradeoff' Spreadsheet
        num_variants: int
            num of variants in the experiment. If control and variant a, then 
            num_variants = 2
        """
        self.monthly_reach = monthly_reach
        self.baseline = baseline
        self.experiment_sizing_timeline = experiment_sizing_timeline
        self.num_variants = num_variants
        self.power = power
        self.alpha = alpha
        self.relative_delta_threshold = relative_threshold
        self.ALTERNATIVE = 'larger'

    def _compute_absolute_delta(self, sample_size: int):
        """ 
        Given power and sample size, compute absolute delta to reach power
        """
        absolute_delta = calculateMetricParameters(baseline=self.baseline, delta=None,
                                                   num_observations=sample_size,
                                                   alpha=self.alpha, power=self.power)
        return absolute_delta

    def _get_additional_deltas_timelines(self, absolute_delta: float) -> pd.DataFrame:
        # --- init variables
        OTHER_RELATIVE_DELTAS_LIST = [
            0.05, 0.08, 0.10, 0.12, 0.14, 0.15, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30]

        # computing the timelines
        other_timelines = []
        for rel_delta in OTHER_RELATIVE_DELTAS_LIST:
            absolute_delta = rel_delta * self.baseline
            sample_needed = calculateMetricParameters(
                baseline=self.baseline, delta=absolute_delta,
                num_observations=None, power=self.power, alpha=self.alpha)
            bucket_size = sample_needed
            timeline_in_months = (
                bucket_size * self.num_variants) / self.monthly_reach
            timeline_in_weeks = timeline_in_months * 4
            other_timelines.append([rel_delta, absolute_delta, bucket_size,
                                    timeline_in_months, timeline_in_weeks])

        # make df
        col_names = ['Relative Delta', 'Absolute Delta',
                     'Bucket Size (sample needed)', 'Months', 'Weeks']
        df_timelines = pd.DataFrame(other_timelines, columns=col_names)

        return df_timelines

    def _get_initial_bucket_size_and_deltas(self):
        """
        Computing the initial bucket size required

        If number of variants > 2, we can reduce timeline by checking if 
        using all variants generate relative delta below 5%
        """
        ALTERNATIVE_DELTA_THRESHOLD = 0.05

        # --- compute bucket size, absolute/relative delta if we use control vs variant a as bucket size
        sample_size_per_bucket = (
            (self.monthly_reach) / 2) / 4 * self.experiment_sizing_timeline
        absolute_delta = self._compute_absolute_delta(
            sample_size=sample_size_per_bucket)
        relative_delta = float(absolute_delta/self.baseline)

        # --- check if using all variants produce a relative delta below 5%
        if self.num_variants > 2:
            alternative_bucket_size = (
                (self.monthly_reach) / self.num_variants) / 4 * self.experiment_sizing_timeline
            alternative_absolute_delta = self._compute_absolute_delta(
                sample_size=alternative_bucket_size)
            alternative_relative_delta = float(
                alternative_absolute_delta/self.baseline)

            if alternative_relative_delta < ALTERNATIVE_DELTA_THRESHOLD:
                sample_size_per_bucket = alternative_absolute_delta
                absolute_delta = alternative_absolute_delta
                relative_delta = alternative_relative_delta

        return sample_size_per_bucket, absolute_delta, relative_delta

    def compute_brief_information(self, ):
        """ 
        Compute all the steps needed to fill in A/B test brief
        Information found in brief table
        --------------------------------
        - reach and baseline
        - suggested delta to detect
        - sample size needed
        - timeline
        Explanation
        -----------
        [ Main Steps ]
            1. Compute initial sample needed:
                - check if we can reduce timeline using all variants (only if relative delta < 5%)
            2. Compute absolute delta -> get_delta(baseline, power, num_obs)
            3. If relative delta >= DELTA_THRESHOLD (12%), perform additional steps to get a more suitable relative delta (additional steps)

            4. Compute Final Timeline
                final_timeline (months) = (sample_size_per_bucket * num_variants) / 
                    monthly_reach

        [ Additional Steps if relative delta >= DELTA_THRESHOLD % ]
            1. 
        """
        # --- Computing deltas using bucket_size calculated from control vs variant a (if initial timeline is respected)
        initial_bucket_size, initial_absolute_delta, initial_relative_delta = self._get_initial_bucket_size_and_deltas()
        final_timeline_in_months = float(
            initial_bucket_size * self.num_variants / self.monthly_reach)
        final_timeline_in_weeks = float(final_timeline_in_months * 4)

        print("")
        print("-------------------------------------------------------------")
        print(f"BASELINE: {self.baseline} ; REACH: {self.monthly_reach}")
        print(
            f"Initial Sample Size (bucket size): {initial_bucket_size}")
        print(
            f"Deltas => absolute: {initial_absolute_delta} ; relative: {initial_relative_delta}")
        print(
            f"Timeline => Months: {final_timeline_in_months} ; Weeks: {final_timeline_in_weeks}")
        print("-------------------------------------------------------------")
        print("")

        # --- Verdict
        need_to_compute_other_timelines = False
        if initial_relative_delta < self.relative_delta_threshold:
            print(
                f"Verdict: We don't need to change the timeline since relative delta is smaller than {self.relative_delta_threshold}")
        else:
            print(
                f"Verdict: We need to change the timeline since relative delta is larger than {self.relative_delta_threshold}")

            need_to_compute_other_timelines = True

        # --- Computing additional timeline if needed
        if need_to_compute_other_timelines:

            df_deltas = self._get_additional_deltas_timelines(
                absolute_delta=initial_absolute_delta)
            print(f"Here are the alternative timelines")
            print('')
            print(df_deltas)




#  -------------------------------------------------------------------------

def test_initial_timeline_okay():
    RELATIVE_DELTA_THRESHOLD = 0.12
    ALPHA, POWER = 0.05, 0.8
    NUM_VARIANTS = 2
    MONTHLY_REACH, BASELINE, EXPERIMENT_SIZING_TIMELINE = 206800, 0.6014, 3


def test_initial_timeline_long():
    RELATIVE_DELTA_THRESHOLD = 0.12
    ALPHA, POWER = 0.05, 0.8
    NUM_VARIANTS = 2
    MONTHLY_REACH, BASELINE, EXPERIMENT_SIZING_TIMELINE = 6883, 0.0486, 3


def main():
    RELATIVE_DELTA_THRESHOLD = 0.12
    ALPHA, POWER = 0.05, 0.8
    NUM_VARIANTS = 3
    MONTHLY_REACH, BASELINE, EXPERIMENT_SIZING_TIMELINE = 90000, 0.0073, 3

    brief = BriefInformation(monthly_reach=MONTHLY_REACH, baseline=BASELINE,
                             experiment_sizing_timeline=EXPERIMENT_SIZING_TIMELINE,
                             num_variants=NUM_VARIANTS, alpha=ALPHA, power=POWER,
                             relative_threshold=RELATIVE_DELTA_THRESHOLD)
    brief.compute_brief_information()


if __name__ == "__main__":
    main()
