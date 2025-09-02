with data as (
    select
	*,
	row_number() over(partition by date, network, platform order by loaded_timestamp desc) as rn
    from {{ ref('stg_fake_data') }}
), clean_data as (
    select
	*
    from data
    where
	rn = 1
)

select
    *
from clean_data
