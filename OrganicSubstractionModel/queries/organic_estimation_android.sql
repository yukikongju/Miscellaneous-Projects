--- cost: 910.9 MB
declare start_date default date '2025-08-01';
declare end_date default date '2025-08-10';

--- DONE: query generate negative conversions for some days ex: '2025-08-03' => are there some days were there are no double counting?


with android_double_counts as (
  select
    -- extract(year from hau_date) as year,
    -- extract(month from hau_date) as month,
    -- extract(week from hau_date) as week,
    date_trunc(hau_date, isoweek) as week_start,
    platform,
    country_code,
    COUNTIF(utm_source = 'google' AND old_hau = 'tvstreaming') as double_count,
    countif(utm_source = 'google') as attribution_count,
  from `relax-melodies-android.late_conversions.users_network_attribution`
  where
    platform = 'android'
    and hau_date >= start_date and hau_date <= end_date
    and hau is not null
    and utm_source is not null
  group by
    -- extract(year from hau_date), extract(month from hau_date), extract(week from hau_date),
    date_trunc(hau_date, isoweek)
    , platform, country_code
), android_double_counting as (
  select
    -- year,
    -- month,
    week_start,
    platform,
    country_code,
    double_count,
    attribution_count,
    case when attribution_count > 0
      then double_count / attribution_count
      else null
    end as double_counting_perc,
  from android_double_counts
), android_af_organic as (
  select
    *
  from `relax-melodies-android.ua_dashboard_prod.pre_final_view`
  where
    date >= start_date and date <= end_date
    and network = 'Organic'
    and platform = 'android'
), android_google as (
  select
    date, platform, country,
    sum(cost_cad) as cost_cad,
    sum(cost_usd) as cost_usd,
    sum(clicks) as clicks,
    sum(impressions) as impressions,
    sum(installs) as installs,
    sum(mobile_trials) as mobile_trials,
    sum(web_trials) as web_trials,
    sum(trials) as trials,
    sum(paid) as paid,
    sum(revenues) as revenues
  from `relax-melodies-android.ua_dashboard_prod.pre_final_view`
  where
    date >= start_date and date <= end_date
    and network = 'googleadwords_int'
    and platform = 'android'
  group by date, platform, country
), android_organic_estimation as (
  select
    af.date,
    af.platform,
    af.network,
    af.country,
    0 as cost_cad,
    0 as cost_usd,
    -- af.impressions - dc.double_counting_perc * google.impressions as impressions,
    -- af.clicks - dc.double_counting_perc * google.clicks as clicks,
    0 as impressions,
    0 as clicks,
    case when af.installs - dc.double_counting_perc * google.installs < 0
	then avg(case when af.installs - dc.double_counting_perc * google.installs > 0 then af.installs - dc.double_counting_perc * google.installs end) over (partition by af.country, af.platform order by af.date rows between 7 preceding and current row)
    else af.installs - dc.double_counting_perc * google.installs end as installs,
    0 as web_trials,
    case when af.mobile_trials - dc.double_counting_perc * google.mobile_trials < 0
	then avg(case when af.mobile_trials - dc.double_counting_perc * google.mobile_trials > 0 then af.mobile_trials - dc.double_counting_perc * google.mobile_trials end) over (partition by af.country, af.platform order by af.date rows between 7 preceding and current row)
    else af.mobile_trials - dc.double_counting_perc * google.mobile_trials end as mobile_trials,
    case when af.trials - dc.double_counting_perc * google.trials < 0
	then avg(case when af.trials - dc.double_counting_perc * google.trials > 0 then af.trials - dc.double_counting_perc * google.trials end) over (partition by af.country, af.platform order by af.date rows between 7 preceding and current row)
    else af.trials - dc.double_counting_perc * google.trials end as trials,
    case when af.paid - dc.double_counting_perc * google.paid < 0
	then avg(case when af.paid - dc.double_counting_perc * google.paid > 0 then af.paid - dc.double_counting_perc * google.paid end) over (partition by af.country, af.platform order by af.date rows between 7 preceding and current row)
    else af.paid - dc.double_counting_perc * google.paid end as paid,
    case when af.revenues - dc.double_counting_perc * google.revenues < 0
	then avg(case when af.revenues - dc.double_counting_perc * google.revenues > 0 then af.revenues - dc.double_counting_perc * google.revenues end) over (partition by af.country, af.platform order by af.date rows between 7 preceding and current row)
    else af.revenues - dc.double_counting_perc * google.revenues end as revenues,
    --  af.paid - dc.double_counting_perc * google.paid as paid,
    --  af.revenues - dc.double_counting_perc * google.revenues as revenues,
    --------

    --  af.installs - dc.double_counting_perc * google.installs as installs,
    --  af.mobile_trials - dc.double_counting_perc * google.mobile_trials as mobile_trials,
    --  af.web_trials - dc.double_counting_perc * google.web_trials as web_trials,
    --  af.trials - dc.double_counting_perc * google.trials as trials,
    --  af.paid - dc.double_counting_perc * google.paid as paid,
    --  af.revenues - dc.double_counting_perc * google.revenues as revenues,
    af.agency,
    af.need_modeling,
    dc.double_counting_perc as double_counting_perc,
  from android_af_organic af
  left join android_google google
    on af.date = google.date
      and af.platform = google.platform
      and af.country = google.country
  left join android_double_counting dc
    on
      -- extract(year from af.date) = dc.year
      -- and extract(month from af.date) = dc.month
      date_trunc(af.date, isoweek) = dc.week_start
      and af.platform = dc.platform
      and af.country = dc.country_code
)

-- , android_organic_estimation_unique as (
--   select
--     *,
--     row_number() over (partition by date, platform, country order by date desc) as rn
--   from android_organic_estimation
-- )


-- select
--   'estimation' as cte,
--   count(*) as count,
-- from android_organic_estimation
-- union all
-- select
--   'af_organic' as cte,
--   count(*) as count
-- from android_af_organic

select *
from android_organic_estimation
where country in ('US', 'CA', 'AU', 'UK', 'HK')
order by country, date

-- select * from android_organic_estimation_unique
-- where
--   rn = 1
--   and installs > 0

-- select * from android_google
-- where country in ('US', 'CA', 'AU', 'UK', 'HK')

-- select * from android_double_counting
-- where
--   country_code = 'US'

-- select * from android_af_organic
-- where
--   country = 'US'
