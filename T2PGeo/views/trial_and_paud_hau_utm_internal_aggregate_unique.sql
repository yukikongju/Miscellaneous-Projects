create or replace view `relax-melodies-android.ua_transform_dev.trial_and_paid_hau_utm_internal_aggregate_unique` as
with aggregates as (
    select *,
    row_number() over (partition by install_date, network, platform, country order by loaded_datetime desc) as rn
    from `relax-melodies-android.ua_transform_dev.trial_and_paid_hau_utm_internal_aggregate` g
)

select
    *
from aggregates
where rn = 1
