create or replace view `relax-melodies-android.ua_transform_dev.geobydate_aggregate_numbers_unique` as
with geobydate_aggregate as (
    select *,
    row_number() over (partition by install_date, network, platform, country order by loaded_datetime desc) as rn
    from `relax-melodies-android.ua_transform_dev.geobydate_aggregate_numbers` g
)

select
    *
from geobydate_aggregate
where rn = 1
