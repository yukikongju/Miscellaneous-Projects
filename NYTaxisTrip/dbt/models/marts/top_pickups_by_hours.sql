{{ config(
    materialized = 'incremental',
    unique_key = ['pickup_date', 'pickup_hour', 'pu_location_id'],
    on_schema_change = 'sync_all_columns',
    indexes = [
        {'columns': ['pickup_date', 'pickup_hour', 'pu_location_id']}
    ]
) }}

with base as (
    select
        pickup_datetime::date as pickup_date,
        extract(hour from pickup_datetime) as pickup_hour,
        pu_location_id,
        count(*) as trip_count
    from {{ ref('fct_trips') }}
    {% if is_incremental() %}
        where pickup_datetime::date > (select max(pickup_date) from {{ this }})
    {% endif %}
    group by 1, 2, 3
)

select * from base
ORDER BY pickup_date, pickup_hour, pu_location_id
