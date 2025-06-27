insert into `relax-melodies-android.test_cumulative_events_table.paid_users_cumulated`

with base_today as (
  select
    snapshot_date,
    num_paid,
    round(avg(num_paid) over (order by snapshot_date rows between 2 preceding and current row), 2) as ma_paid_3d,
    current_timestamp() as created_at
  from `relax-melodies-android.test_cumulative_events_table.paid_users_daily`
  where snapshot_date between DATE_SUB(current_date(), INTERVAL 2 DAY) and current_date()
), today as (
  select
    snapshot_date,
    num_paid,
    ma_paid_3d,
    created_at
  from base_today
  where snapshot_date = current_date()
)

select * from today
