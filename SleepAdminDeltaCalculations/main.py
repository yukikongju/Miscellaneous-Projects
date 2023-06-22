import pandas as pd

from statsmodels.stats.power import zt_ind_solve_power


class BriefInformation:


    def __init__(self, monthly_reach: int, baseline: float, experiment_sizing_timeline: int, num_variants: int, POWER = 0.8, ALPHA = 0.05, RELATIVE_DELTA_THRESHOLD = 0.12):
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
        self.POWER = POWER
        self.ALPHA = ALPHA
        self.RELATIVE_DELTA_THRESHOLD = RELATIVE_DELTA_THRESHOLD
        self.ALTERNATIVE = 'larger'


    def _compute_absolute_delta(self, sample_size: int): # FIXME
        """ 
        Given power and sample size, compute absolute delta to reach power
        """
        absolute_delta = zt_ind_solve_power(effect_size=None, nobs1=sample_size,
                                            alpha=self.ALPHA, power=self.POWER,
                                            ratio=1.0, alternative=self.ALTERNATIVE)
        return absolute_delta


    def _get_additional_deltas_timelines(self, absolute_delta: float) -> pd.DataFrame: # TODO
        # --- init variables
        OTHER_RELATIVE_DELTAS_LIST = [0.5, 0.8, 0.10, 0.12, 0.14, 0.15, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30]

        other_timelines = []
        pass


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
            1. Compute initial sample needed: regardless of number
                initial_sample_size = (monthly_reach / 2 variants) / 4 weeks * experiment_sizing_timeline
            2. Compute absolute delta -> get_delta(baseline, power, num_obs)
            3. If relative delta >= 10%, perform additional steps to get a more suitable relative delta (additional steps)
            4. Compute Final Timeline
                final_timeline (months) = (sample_size_per_bucket * num_variants) / 
                    monthly_reach

        [ Additional Steps if relative delta >= 10% ]
            1. 


        """


        # --- Computing deltas, bucket_size if initial timeline is respected
        initial_sample_size_per_bucket = ((self.monthly_reach) / 2) / 4 * self.experiment_sizing_timeline
        initial_absolute_delta = self._compute_absolute_delta(sample_size=initial_sample_size_per_bucket)
        initial_relative_delta = float(initial_absolute_delta / self.baseline)
        final_timeline_in_months = float(initial_sample_size_per_bucket * self.num_variants / self.monthly_reach)
        final_timeline_in_weeks = float(final_timeline_in_months * 4)

        print("")
        print("-------------------------------------------------------------")
        print(f"BASELINE: {self.baseline} ; REACH: {self.monthly_reach}")
        print(f"Initial Sample Size: {initial_sample_size_per_bucket}")
        print(f"Deltas => absolute: {initial_absolute_delta} ; relative: {initial_relative_delta}")
        print(f"Timeline => Months: {final_timeline_in_months} ; Weeks: {final_timeline_in_weeks}")
        print("-------------------------------------------------------------")
        print("")

        # --- Verdict
        need_to_compute_other_timelines = False
        if initial_relative_delta < self.RELATIVE_DELTA_THRESHOLD:
            print(f"Verdict: We don't need to change the timeline since relative delta smaller than {self.RELATIVE_DELTA_THRESHOLD}")
        else:
            need_to_compute_other_timelines = True

        # --- Computing additional timeline if needed
        if need_to_compute_other_timelines:

            df_deltas = self._get_additional_deltas_timelines(absolute_delta=initial_absolute_delta)
            print(f"Here is the alternative timelines")
            print(df_deltas)



def main():
    MONTHLY_REACH, BASELINE, EXPERIMENT_SIZING_TIMELINE = 6683, 0.0486, 3
    NUM_VARIANTS = 2
    POWER = 0.8
    ALPHA = 0.5

    brief = BriefInformation(monthly_reach=MONTHLY_REACH, baseline=BASELINE,
                             experiment_sizing_timeline=EXPERIMENT_SIZING_TIMELINE,
                             num_variants=2)
    brief.compute_brief_information()
    

if __name__ == "__main__":
    main()
