create table if not exists `relax-melodies-android.test_cumulative_events_table.paid_users_cumulated` (
    snapshot_date date,
    num_paid int64,
    ma_paid_3d float64,
    created_at timestamp,
)
