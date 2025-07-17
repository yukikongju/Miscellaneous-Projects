create or replace table `relax-melodies-android.late_conversions.users_paid_attribution` (
    user_id STRING,
    user_pseudo_id STRING,
    platform STRING,
    country STRING,
    campaign STRING,
    hau STRING,
    utm STRING,
    trial_event_date DATE,
    paid_event_date DATE,
)
