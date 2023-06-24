import numpy as np
#  import statsmodels.stats.proportion as smp

import statsmodels.stats.proportion as prop
from statsmodels.stats.power import tt_ind_solve_power, zt_ind_solve_power


def calculateMetricParameters(baseline: float, delta: float, num_observations: int, power=0.8, alpha=0.05):
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
        delta = np.sin(np.arcsin(np.sqrt(baseline)) -
                       effect_size / 2) ** 2 - baseline
        delta = float(np.abs(delta))

        return delta

    def ind_solve_power(baseline: float, delta: float, nobs1: int, alpha: float, power: float, ratio: float,
                        alternative: str) -> float:
        if delta is not None:
            effect_size = get_prop_effect_size(baseline=baseline, delta=delta)
        else:
            effect_size = None

        new_param = zt_ind_solve_power(effect_size=effect_size, nobs1=nobs1, alpha=alpha, power=power,
                                       ratio=ratio, alternative=alternative)

        if delta is None:
            new_param = get_prop_effect_size_invert(
                baseline=baseline, effect_size=new_param)

        return new_param

    def validate_inputs(baseline: float, delta: float, nobs1: float, alpha: float, power: float) -> None:
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

    ALTERNATIVE = 'smaller'
    RATIO = 1.0

    validate_inputs(baseline, delta, num_observations, alpha, power)

    new_param = ind_solve_power(baseline=baseline, delta=delta, nobs1=num_observations, alpha=alpha, power=power,
                                ratio=RATIO, alternative=ALTERNATIVE)

    return new_param

