with data as (
    select * from {{ source('raw', 'fake_data') }}
), clean_data as (
    select * from data
)

select
    *
from clean_data
