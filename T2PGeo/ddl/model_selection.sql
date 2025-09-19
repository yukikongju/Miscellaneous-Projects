--- create
create or replace table `relax-melodies-android.ua_transform_prod.model_selection` (
    platform string,
    network string,
    model_source string,
    start_date datetime,
    end_date datetime,
    updated_at timestamp default current_timestamp()
)
cluster by platform, network;

--- T2P default geobydate
insert into `relax-melodies-android.ua_transform_prod.model_selection` (
    platform, network, model_source, start_date, end_date, updated_at
)
values
    ('ios', 'Apple Search Ads', 'Geobydate', '2000-01-01', '2999-12-31', current_timestamp()),
    ('android', 'Facebook Ads', 'Geobydate', '2000-01-01', '2999-12-31', current_timestamp()),
    ('ios', 'Facebook Ads', 'Geobydate', '2000-01-01', '2999-12-31', current_timestamp()),
    ('android', 'googleadwords_int', 'Internal', '2000-01-01', '2999-12-31', current_timestamp()),
    ('ios', 'googleadwords_int', 'Internal', '2000-01-01', '2999-12-31', current_timestamp()),
    ('ios', 'tiktokglobal_int', 'Geobydate', '2000-01-01', '2999-12-31', current_timestamp()),
    ('android', 'tiktokglobal_int', 'Geobydate', '2000-01-01', '2999-12-31', current_timestamp()),
    ('ios', 'snapchat_int', 'Geobydate', '2000-01-01', '2999-12-31', current_timestamp()),
    ('android', 'snapchat_int', 'Geobydate', '2000-01-01', '2999-12-31', current_timestamp()),
    ('ios', 'tatari_linear', 'Internal', '2000-01-01', '2999-12-31', current_timestamp()),
    ('android', 'tatari_linear', 'Internal', '2000-01-01', '2999-12-31', current_timestamp()),
    ('ios', 'tatari_streaming', 'Internal', '2000-01-01', '2999-12-31', current_timestamp()),
    ('android', 'tatari_streaming', 'Internal', '2000-01-01', '2999-12-31', current_timestamp()),
    ('ios', 'tatari_programmatic', 'Internal', '2000-01-01', '2999-12-31', current_timestamp()),
    ('android', 'tatari_programmatic', 'Internal', '2000-01-01', '2999-12-31', current_timestamp());
