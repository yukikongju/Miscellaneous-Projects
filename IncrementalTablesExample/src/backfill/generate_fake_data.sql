--- cost: 0B for a single day
declare start_date date default '2025-08-01';
declare end_date date default '2025-09-01';
declare num_days_back int64 default 45;
declare networks array<string> default ['Facebook Ads', 'googleadwords_int']; -- to change
declare platforms array<string> default ['ios', 'android'];
declare countries array<string> default ['US', 'CA', 'AU', 'FR', 'UK'];


FOR rec IN (
  SELECT current_day
  FROM UNNEST(GENERATE_DATE_ARRAY(start_date, end_date, INTERVAL 1 DAY)) AS current_day
) DO
BEGIN

INSERT INTO `relax-melodies-android.test_incremental.fake_data`

with generated_timestamp as (
  select TIMESTAMP(DATETIME(
    rec.current_day,
    TIME(
      CAST(RAND() * 23 AS INT64),
      CAST(RAND() * 60 AS INT64),
      CAST(RAND() * 60 AS INT64)
    ))) as ts
)

select
    day,
    platform,
    network,
    country,
    floor((1 / (1 + exp(-3.14 * (date_diff(rec.current_day, day, DAY) / num_days_back)))) * RAND() * 2000) as cost,
    floor((1 / (1 + exp(-3.14 * (date_diff(rec.current_day, day, DAY) / num_days_back)))) * RAND() * 10000) as impressions,
    floor((1 / (1 + exp(-3.14 * (date_diff(rec.current_day, day, DAY) / num_days_back)))) * RAND() * 5000) as clicks,
    floor((1 / (1 + exp(-3.14 * (date_diff(rec.current_day, day, DAY) / num_days_back)))) * RAND() * 1000) as installs,
    floor((1 / (1 + exp(-3.14 * (date_diff(rec.current_day, day, DAY) / num_days_back)))) * RAND() * 200) as trials,
    floor((1 / (1 + exp(-3.14 * (date_diff(rec.current_day, day, DAY) / num_days_back)))) * RAND() * 100) as paid,
    0 as revenue,
    g.ts as loaded_timestamp,
    from unnest(GENERATE_DATE_ARRAY(DATE_SUB(rec.current_day, INTERVAL num_days_back DAY), rec.current_day, interval 1 day)) as day
  cross join unnest(platforms) as platform
  cross join unnest(networks) as network
  cross join unnest(countries) as country
  cross join generated_timestamp g;

END;
END FOR;
