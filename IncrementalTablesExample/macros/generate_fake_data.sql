{% macro generate_fake_data(start_date, end_date, loaded_datetime, platforms, networks) %} (
    select
	day,
	platform,
	network,
	floor(RAND() * 10000) as impressions,
	floor(RAND() * 5000) as clicks,
	floor(RAND() * 1000) as installs,
	floor(RAND() * 200) as trials,
	floor(RAND() * 100) as paid,
	{{ loaded_datetime }} as loaded_datetime,
	from unnest(GENERATE_DATE_ARRAY({{ start_date }}, {{ end_date }}, interval 1 day)) as day
      cross join unnest({{ platforms }}) as platform
      cross join unnest({{ networks }}) as network
)
{% endmacro %}
