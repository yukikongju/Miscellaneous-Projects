{{
config(
    materialized="incremental",
    cluster_by=["network", "platform", "country"],
    partition_by={
	"field": "date",
	"data_type": "date",
	"granularity": "day"
    },
    incremental_strategy = 'insert_overwrite'
)
}}

with fake_data as (
    select
	*
    from {{ source('raw', 'fake_data') }}
    {% if is_incremental() %}
	where loaded_timestamp > (select max(loaded_timestamp) from {{ this }})
    {% endif %}
)

select * from fake_data
