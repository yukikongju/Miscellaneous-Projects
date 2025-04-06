select
    distinct payment_type as payment_id,
    case
	when payment_type = 0 then 'Flex Fare Trip'
	when payment_type = 1 then 'Credit Card'
	when payment_type = 2 then 'Cash'
	when payment_type = 3 then 'No Charge'
	when payment_type = 4 then 'Dispute'
	when payment_type = 5 then 'Unknown'
	when payment_type = 6 then 'Voided'
	--  else 'Error'
    end as payment_name
from {{ ref('stg_yellow_taxis') }}
order by payment_id
