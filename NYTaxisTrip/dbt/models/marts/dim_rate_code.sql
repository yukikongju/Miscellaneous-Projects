select
    distinct rate_code_id,
    case
	when rate_code_id = 1 then 'Standard Rate'
	when rate_code_id = 2 then 'JFK'
	when rate_code_id = 3 then 'Newark'
	when rate_code_id = 4 then 'Nassauu or Westchester'
	when rate_code_id = 5 then 'Negociated Fare'
	when rate_code_id = 6 then 'Group Ride'
	when rate_code_id = 99 then 'Null/Unknown'
	--  else 'Error'
    end as rate_code_name
from {{ ref('stg_yellow_taxis') }}
order by rate_code_id
