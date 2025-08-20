select
  extract(year from date) as year,
  extract(month from date) as month,
  extract(week from date) as week,
  -- case when sum(total_cost) > 0
  --   then sum(total_cost * null_conversion_value_rate) / sum(total_cost)
  --   else null
  -- end as weighted_null_conversion_rate,
  avg(null_conversion_value_rate) as avg_null_conversion_rate,
from `relax-melodies-android.ua_extract_prod.appsflyer_skan_modeled`
where
  media_source_pid = 'googleadwords_int'
  and null_conversion_value_rate is not null
  -- and total_cost > 0
  and campaign_c like '%US%'
  -- and date > '2024-07-01'
group by
  extract(year from date), extract(month from date), extract(week from date)
order by
  extract(year from date), extract(month from date), extract(week from date)
