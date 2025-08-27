declare start_date default date '2025-08-01';
declare end_date default date '2025-09-01';


with daily_table as (
  select
    daily_startdate,
    date_trunc(daily_startdate, ISOWEEK) as isoweek,
    1 as value
  from
    unnest(generate_date_array(start_date, end_date, interval 1 day)) as daily_startdate
), weekly_table as (
    select
    date_trunc(weekly_startdate, ISOWEEK) as weekly_startdate,
    2 as value
  from
    unnest(generate_date_array(start_date, end_date, interval 7 day)) as weekly_startdate
), daily_aggregate as (
  select
    dt.daily_startdate as date,
    dt.value,
    wt.value
  from daily_table dt
  join weekly_table wt
    on date_trunc(dt.daily_startdate, ISOWEEK) = wt.weekly_startdate
)

select * from daily_aggregate

-- select * from daily_table
-- select * from weekly_table
