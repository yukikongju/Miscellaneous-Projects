select
    distinct taxi_type_id,
    case
	when taxi_type_id = 1 then 'Yellow'
	when taxi_type_id = 2 then 'Green'
    end as taxi_type_name
from {{ ref('fact_trips') }}
