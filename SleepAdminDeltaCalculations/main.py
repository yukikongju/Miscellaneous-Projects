import pandas as pd
import numpy as np

from utils import compute_weekly_reach, compute_deltas
from utils import compute_absolute_delta_from_relative_delta, compute_sample_needed

def compute_timeline_deltas(monthly_reach: float, conversion_rate: float, power = 0.8, alpha = 0.5, num_weeks = 15, num_variants=2):
    """
    Compute absolute and relative deltas for week 1 to <num_weeks>

    Parameters
    ----------
    conversion_rate: float
        percentage between 0 and 1

    """
    weekly_reach = compute_weekly_reach(monthly_reach, num_variants)

    # --- compute absolute and relative deltas for every week
    timeline_results = []
    for week in range(1, num_weeks+1):
        sample_size_needed = week * weekly_reach
        absolute_delta, relative_delta = compute_deltas(sample_size=sample_size_needed, conversion_rate=conversion_rate)
        timeline_results.append([week, sample_size_needed, absolute_delta, relative_delta])

    # --- into dataframe
    col_names = ['Week', 'Sample Size', 'Absolute Delta', 'Relative Delta']
    df = pd.DataFrame(timeline_results, columns=col_names)

    return df

    
def compute_deltas_timelines(monthly_reach: float, conversion_rate: float, until_relative_delta: float,  power = 0.8, alpha = 0.5, num_variants=2):
    """ 
    Compute timeline to reach relative deltas

    Parameters
    ----------
    until_relative_delta: float
        compute time for each delta until this relative delta

    """
    weekly_reach = compute_weekly_reach(monthly_reach, num_variants)
    #  relative_deltas = [ float(i) for i in range(until_relative_delta) ]
    relative_deltas = np.arange(0.05, until_relative_delta+0.05, 0.05).tolist()
    print(relative_deltas)

    # --- compute num of weeks needed for each relative delta
    timeline_results = []
    for relative_delta in relative_deltas:
        absolute_delta = compute_absolute_delta_from_relative_delta(relative_delta, conversion_rate)
        sample_size_needed = compute_sample_needed(conversion_rate, absolute_delta)
        num_week_needed = float(sample_size_needed / weekly_reach)
        num_months_needed = float(num_week_needed / 4)
        timeline_results.append([relative_delta, absolute_delta, sample_size_needed, num_week_needed, num_months_needed])


    # --- into dataframe
    col_names = ['Relative Delta', 'Absolute Delta', 'Sample Size Needed', 'Num of Weeks', 'Num of Months']
    df = pd.DataFrame(timeline_results, columns=col_names)

    return df


def main():
    METRIC_NAME = ''
    MONTHLY_REACH, CONVERSION_RATE = 6883, 0.0486

    print(compute_timeline_deltas(monthly_reach=MONTHLY_REACH, conversion_rate=CONVERSION_RATE))
    print(compute_deltas_timelines(monthly_reach=MONTHLY_REACH, conversion_rate=CONVERSION_RATE, until_relative_delta=0.30))

    

if __name__ == "__main__":
    main()
