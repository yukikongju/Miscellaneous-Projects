with rates_ma as (
  select
    paid_year_month,
    platform,
    round(avg(`1-Year`) over (partition by platform order by paid_year_month rows between 2 preceding and current row), 2) as `1-Year`,
    round(avg(`2-Years`) over (partition by platform order by paid_year_month rows between 2 preceding and current row), 2) as `2-Years`,
    round(avg(`3-Years`) over (partition by platform order by paid_year_month rows between 2 preceding and current row), 2) as `3-Years`,
    round(avg(`4-Years`) over (partition by platform order by paid_year_month rows between 2 preceding and current row), 2) as `4-Years`,
    round(avg(`5-Years`) over (partition by platform order by paid_year_month rows between 2 preceding and current row), 2) as `5-Years`,
    round(avg(`>5-Years`) over (partition by platform order by paid_year_month rows between 2 preceding and current row), 2) as `>5-Years`,
  from `relax-melodies-android.late_conversions.renewal_rates_cohorted`
), last_valid_rates as (
  select
    platform,
    last_value(`1-Year`) over (
      partition by platform
      order by case
        when parse_date('%Y-%m', paid_year_month) < date_sub(CURRENT_DATE(), interval 1 YEAR) then paid_year_month end
    rows between UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as `latest_1Year_renewal_rate`,
    last_value(`2-Years`) over (
      partition by platform
      order by case
        when parse_date('%Y-%m', paid_year_month) < date_sub(CURRENT_DATE(), interval 2 YEAR) then paid_year_month end
    rows between UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as `latest_2Year_renewal_rate`,
    last_value(`3-Years`) over (
      partition by platform
      order by case
        when parse_date('%Y-%m', paid_year_month) < date_sub(CURRENT_DATE(), interval 3 YEAR) then paid_year_month end
    rows between UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as `latest_3Year_renewal_rate`,
    last_value(`4-Years`) over (
      partition by platform
      order by case
        when parse_date('%Y-%m', paid_year_month) < date_sub(CURRENT_DATE(), interval 4 YEAR) then paid_year_month end
    rows between UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as `latest_4Year_renewal_rate`,
    last_value(`5-Years`) over (
      partition by platform
      order by case
        when parse_date('%Y-%m', paid_year_month) < date_sub(CURRENT_DATE(), interval 5 YEAR) then paid_year_month end
    rows between UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as `latest_5Year_renewal_rate`,
    last_value(`>5-Years`) over (
      partition by platform
      order by case
        when parse_date('%Y-%m', paid_year_month) < date_sub(CURRENT_DATE(), interval 6 YEAR) then paid_year_month end
    rows between UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as `latest_5Year_renewal_rate`,
  from rates_ma
  qualify row_number() over (partition by platform order by paid_year_month ASC) = 1
)

select
  *
from last_valid_rates
