-- models/marts/core/dim_datetime.sql

WITH full_dt AS (
    {{ dbt_date.get_base_dates(start_date="2024-01-01", end_date="2024-02-01", datepart="day") }}
),

-- Combine base dates with time zone and other enrichments
        --  {{ dbt_date.convert_timezone("f.date_day", "America/New_York", "UTC") }} AS date_in_utc,
        --  {{ dbt_date.convert_timezone("f.date_day", "America/New_York", source_tz="UTC") }} AS date_from_utc,
        --  {{ dbt_date.convert_timezone("f.date_day", "America/New_York") }} AS date_local,
full_dt_tr AS (
    SELECT
        f.date_day AS full_date,
	{{ dbt_date.day_name("f.date_day", short=True) }} as day_of_week_short_name,
	{{ dbt_date.day_name("f.date_day", short=False) }} as day_of_week_long_name,
	{{ dbt_date.day_of_month("f.date_day") }} as day_of_month,
	{{ dbt_date.day_of_year("f.date_day") }} as day_of_year,
	{{ dbt_date.iso_week_end("f.date_day") }} as iso_week_end_date,
	{{ dbt_date.month_name("f.date_day", short=true) }} as month_short_name,
	{{ dbt_date.month_name("f.date_day", short=false) }} as month_long_name,
        f.date_day::timestamp AS ts_direct,
        f.date_day::timestamp AT TIME ZONE 'UTC' AS ts_utc
    FROM full_dt as f
)

SELECT
    {{ dbt_utils.generate_surrogate_key(['ts_direct']) }} AS date_key,
    *
FROM full_dt_tr
