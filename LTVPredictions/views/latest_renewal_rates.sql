with cohorts_ranked as (
  select
    format_date('%Y-%m', today) as year_month,
    *,
  rank() over (partition by network, platform, country_code, renewal_bucket order by paid_year_month desc) as rn
  from `relax-melodies-android.late_conversions.mature_renewal_cohorts`
  --  where
  --    ((renewal_bucket = '1-Year' AND paid_year_month < FORMAT_DATE('%Y-%m', DATE_SUB(today, INTERVAL 1 year)))
  --    OR (renewal_bucket = '2-Years' AND paid_year_month < FORMAT_DATE('%Y-%m', DATE_SUB(today, INTERVAL 2 year)))
  --    OR (renewal_bucket = '3-Years' AND paid_year_month < FORMAT_DATE('%Y-%m', DATE_SUB(today, INTERVAL 3 year)))
))

select
  year_month,
  network,
  platform,
  country_code,
  renewal_bucket,
  num_renewals,
  num_paid,
  paid_proceeds,
  renewal_proceeds,
  renewal_percentage
from cohorts_ranked
where rn = 1
