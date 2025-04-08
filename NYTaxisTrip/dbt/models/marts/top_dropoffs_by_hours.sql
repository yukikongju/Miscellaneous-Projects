{{ config(
    materialized = 'incremental',
    unique_key = ['dropoff_date', 'dropoff_hour', 'do_location_id'],
    on_schema_change = 'sync_all_columns',
    indexes = [
        {'columns': ['dropoff_date', 'dropoff_hour', 'do_location_id']}
    ]
) }}

with base as (
    select
        dropoff_datetime::date as dropoff_date,
        extract(hour from dropoff_datetime) as dropoff_hour,
        do_location_id,
        count(*) as trip_count
    from {{ ref('fct_trips') }}
    {% if is_incremental() %}
        where dropoff_datetime::date > (select max(dropoff_date) from {{ this }})
    {% endif %}
    group by 1, 2, 3
)

select * from base
ORDER BY dropoff_date, dropoff_hour, do_location_id
