import statsmodels.stats.proportion as smp

from statsmodels.stats.power import tt_ind_solve_power

def compute_deltas(sample_size: int, conversion_rate: float, power=0.8, alpha=0.5):
    """
    Compute absolute and relative deltas given sample size and conversion_rate 
    (baseline)

    doc: https://www.statsmodels.org/stable/generated/statsmodels.stats.power.tt_ind_solve_power.html

    """
    # --- TODO: verify if ratio and effect_size are okay

    # --- absolute delta calculation: using 'larger' instead of 'two-sided'
    absolute_delta = tt_ind_solve_power(effect_size=None, nobs1=sample_size,
                                        alpha=alpha, power=power, ratio=1.0, 
                                        alternative='larger')

    absolute_delta *= conversion_rate

    # --- relative delta calculation
    relative_delta = float(absolute_delta / conversion_rate)

    return absolute_delta, relative_delta


def compute_weekly_reach(monthly_reach: int, num_variants: int):
    """
    Calculate the weekly reach that goes in each variant bucket

    Note: if experiment has control and variant a, num_variants = 2
    """
    return (monthly_reach / num_variants) / 4
    

def compute_sample_needed(conversion_rate: float, absolute_delta: float, power=0.8, alpha=0.5):
    """ 
    Compute sample needed to reach a given absolute delta
    """
    sample_size =  smp.samplesize_proportions_2indep_onetail(diff=absolute_delta, prop2=conversion_rate, power=power)

    return sample_size

#  ---------------------------------------------------------------------------


def compute_absolute_delta_from_relative_delta(relative_delta: float, conversion_rate: float):
    """ 
    Compute absolute delta from relative delta
    """
    absolute_delta = relative_delta * conversion_rate
    return absolute_delta


#  ---------------------------------------------------------------------------


def test_compute_deltas():
    print(compute_deltas(6883, 0.0486))



def tests():
    test_compute_deltas()
    

#  ---------------------------------------------------------------------------


if __name__ == "__main__":
    tests()


