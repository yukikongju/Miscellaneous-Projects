import pandas as pd
import numpy as np
import statsmodels.stats.proportion as prop

from statsmodels.stats.power import zt_ind_solve_power


class BriefInformation:

    def __init__(
        self,
        monthly_reach: int,
        baseline: float,
        experiment_sizing_timeline: int,
        num_variants: int,
        power=0.8,
        alpha=0.05,
        relative_threshold=0.1,
    ):
        """
        Arguments
        ---------
        monthly_reach: int
            number of user reached. This information is found in MixPanel
        baseline: float
            conversion rate between 0 and 1. This information is found in
            MixPanel
        experiment_sizing_timeline: int
            week neededed to complete this experiment in weeks. This
            information can be found in the 'Experiment Sizing Tradeoff'
            Spreadsheet
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
        self.ALTERNATIVE = "larger"

    def _compute_absolute_delta(self, sample_size: int):
        """
        Given power and sample size, compute absolute delta to reach power
        """
        absolute_delta = calculateMetricParameters(
            baseline=self.baseline,
            delta=None,
            num_observations=sample_size,
            alpha=self.alpha,
            power=self.power,
        )
        return absolute_delta

    def _get_additional_timelines(
        self,
    ):
        OTHER_WEEKS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        results = []
        for n_week in OTHER_WEEKS:
            sample_size_per_bucket = int(self.monthly_reach / self.num_variants) / 4 * n_week
            absolute_delta = self._compute_absolute_delta(sample_size=sample_size_per_bucket)
            relative_delta = float(absolute_delta / self.baseline)
            res = [n_week, relative_delta, absolute_delta, sample_size_per_bucket]
            results.append(res)

        df = pd.DataFrame(
            results, columns=["Weeks", "Relative Delta", "Absolute Delta", "Sample Size per Bucket"]
        )
        return df

    def _get_additional_deltas_timelines(self, absolute_delta: float) -> pd.DataFrame:
        """
        DEPRECATED
        """
        # --- init variables
        OTHER_RELATIVE_DELTAS_LIST = [
            0.05,
            0.06,
            0.07,
            0.08,
            0.09,
            0.10,
            0.11,
            0.12,
            0.13,
            0.14,
            0.15,
            0.16,
            0.17,
            0.18,
            0.20,
            0.22,
            0.24,
            0.26,
            0.28,
            0.30,
        ]

        # computing the timelines
        other_timelines = []
        for rel_delta in OTHER_RELATIVE_DELTAS_LIST:
            absolute_delta = rel_delta * self.baseline
            sample_needed = calculateMetricParameters(
                baseline=self.baseline,
                delta=absolute_delta,
                num_observations=None,
                power=self.power,
                alpha=self.alpha,
            )
            bucket_size = sample_needed
            timeline_in_months = (bucket_size * self.num_variants) / self.monthly_reach
            timeline_in_weeks = timeline_in_months * 4
            other_timelines.append(
                [rel_delta, absolute_delta, bucket_size, timeline_in_months, timeline_in_weeks]
            )

        # make df
        col_names = [
            "Relative Delta",
            "Absolute Delta",
            "Bucket Size (sample needed)",
            "Months",
            "Weeks",
        ]
        df_timelines = pd.DataFrame(other_timelines, columns=col_names)

        return df_timelines

    def _get_initial_bucket_size_and_deltas(self):
        """
        Computing the initial bucket size required

        If number of variants > 2, we can reduce timeline by checking if
        using all variants generate relative delta below 5%

        Per Hamed recommendation: always use all variants when determining
        bucket size
        """
        #  ALTERNATIVE_DELTA_THRESHOLD = 0.05

        # --- compute bucket size, absolute/relative delta if we use control vs variant a as bucket size
        #  sample_size_per_bucket = (
        #      (self.monthly_reach) / 2) / 4 * self.experiment_sizing_timeline
        sample_size_per_bucket = (
            ((self.monthly_reach) / self.num_variants) / 4 * self.experiment_sizing_timeline
        )
        absolute_delta = self._compute_absolute_delta(sample_size=sample_size_per_bucket)
        relative_delta = float(absolute_delta / self.baseline)

        # --- check if using all variants produce a relative delta below 5%
        #  if self.num_variants > 2:
        #      alternative_bucket_size = (
        #          (self.monthly_reach) / self.num_variants) / 4 * self.experiment_sizing_timeline
        #      alternative_absolute_delta = self._compute_absolute_delta(
        #          sample_size=alternative_bucket_size)
        #      alternative_relative_delta = float(
        #          alternative_absolute_delta/self.baseline)

        #      if alternative_relative_delta < ALTERNATIVE_DELTA_THRESHOLD:
        #          sample_size_per_bucket = alternative_bucket_size
        #          absolute_delta = alternative_absolute_delta
        #          relative_delta = alternative_relative_delta

        return sample_size_per_bucket, absolute_delta, relative_delta

    def compute_brief_information(
        self,
    ):
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
        initial_bucket_size, initial_absolute_delta, initial_relative_delta = (
            self._get_initial_bucket_size_and_deltas()
        )
        initial_timeline_in_months = float(
            initial_bucket_size * self.num_variants / self.monthly_reach
        )
        initial_timeline_in_weeks = float(initial_timeline_in_months * 4)

        print("")
        print("-------------------------------------------------------------")
        print(f"BASELINE: {self.baseline} ; REACH: {self.monthly_reach}")
        print(f"Initial Sample Size (bucket size): {initial_bucket_size}")
        print(f"Deltas => absolute: {initial_absolute_delta} ; relative: {initial_relative_delta}")
        print(
            f"Timeline => Months: {initial_timeline_in_months} ; Weeks: {initial_timeline_in_weeks}"
        )
        print("-------------------------------------------------------------")
        print("")

        # --- Verdict
        need_to_compute_other_timelines = False
        if initial_relative_delta < self.relative_delta_threshold:
            print(
                f"Verdict: We don't need to change the timeline since relative delta is smaller than {self.relative_delta_threshold}"
            )
        else:
            print(
                f"Verdict: We need to change the timeline since relative delta is larger than {self.relative_delta_threshold}"
            )

            need_to_compute_other_timelines = True

        # --- Computing additional timeline if needed
        if need_to_compute_other_timelines:

            df_deltas = self._get_additional_deltas_timelines(absolute_delta=initial_absolute_delta)
            print(f"Here are the alternative timelines")
            print("")
            print(df_deltas)

        return (
            initial_bucket_size,
            initial_absolute_delta,
            initial_relative_delta,
            initial_timeline_in_months,
            initial_timeline_in_weeks,
        )


def calculateMetricParameters(
    baseline: float, delta: float, num_observations: int, power=0.8, alpha=0.05
):
    """
    This Function is used in Sleep Admin to compute the following:
    Given baseline and power, compute:
    - Number of observations if delta is given
    - delta if number of observations is given
    """

    def get_prop_effect_size(baseline: float, delta: float) -> float:
        variant = baseline + delta

        std_effect = prop.proportion_effectsize(prop1=baseline, prop2=variant)
        std_effect = float(std_effect)

        return std_effect

    def get_prop_effect_size_invert(baseline: float, effect_size: float) -> float:
        delta = np.sin(np.arcsin(np.sqrt(baseline)) - effect_size / 2) ** 2 - baseline
        delta = float(np.abs(delta))

        return delta

    def ind_solve_power(
        baseline: float,
        delta: float,
        nobs1: int,
        alpha: float,
        power: float,
        ratio: float,
        alternative: str,
    ) -> float:
        if delta is not None:
            effect_size = get_prop_effect_size(baseline=baseline, delta=delta)
        else:
            effect_size = None

        new_param = zt_ind_solve_power(
            effect_size=effect_size,
            nobs1=nobs1,
            alpha=alpha,
            power=power,
            ratio=ratio,
            alternative=alternative,
        )

        if delta is None:
            new_param = get_prop_effect_size_invert(baseline=baseline, effect_size=new_param)

        return new_param

    def validate_inputs(
        baseline: float, delta: float, nobs1: float, alpha: float, power: float
    ) -> None:
        def if_exist_and_not_ratio_raise_error(param_name: str, value: float) -> bool:

            if value:
                if not 0 < alpha < 1:
                    raise ValueError(f"{param_name} is not between 0 and 1")

        if_exist_and_not_ratio_raise_error("delta", delta)
        if_exist_and_not_ratio_raise_error("alpha", alpha)
        if_exist_and_not_ratio_raise_error("power", power)

        if baseline is None:
            raise ValueError("Baseline should be defined")

        if delta:
            if 1 <= baseline + delta:
                raise ValueError("Delta + baseline is greater or equal than 1")

        if nobs1:
            if nobs1 < 0:
                raise ValueError("num_observations is not greater than 1")

    ALTERNATIVE = "smaller"
    RATIO = 1.0

    validate_inputs(baseline, delta, num_observations, alpha, power)

    new_param = ind_solve_power(
        baseline=baseline,
        delta=delta,
        nobs1=num_observations,
        alpha=alpha,
        power=power,
        ratio=RATIO,
        alternative=ALTERNATIVE,
    )

    return new_param
