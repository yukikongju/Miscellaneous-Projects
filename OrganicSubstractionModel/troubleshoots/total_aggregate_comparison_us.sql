select
*
from `relax-melodies-android.organics.total_aggregate_comparison`
where
  country = 'US'
  and date >= '2024-01-01'
order by
  platform, country, date
