--  relax-melodies-android.ua_transform_dev.trial2paid_geobydate_unique
with geobydate_t2p as (
    select *,
    row_number() over (partition by install_date, network, platform, country order by extracted_datetime desc) as rn
    from `relax-melodies-android.ua_transform_prod.trial2paid_geobydate_model` g
)

select
    *
from geobydate_t2p
where rn = 1
