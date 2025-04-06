with seconds as (
    select generate_series(0, 86399) as second_of_day
), dim_time as (
    select
	second_of_day,
        make_time(
            second_of_day / 3600,
            (second_of_day % 3600) / 60,
            second_of_day % 60
        ) as time_of_day,

        -- Useful breakdowns
        (second_of_day / 3600)::int as hour,
        ((second_of_day % 3600) / 60)::int as minute,
        (second_of_day % 60)::int as second,

        -- Formatted string (e.g., 13:04:05)
        to_char(make_time(
            second_of_day / 3600,
            (second_of_day % 3600) / 60,
            second_of_day % 60
        ), 'HH24:MI:SS') as time_string

    from seconds
)

select
    {{ dbt_utils.generate_surrogate_key(['second_of_day']) }} as time_key,
    *
from dim_time
