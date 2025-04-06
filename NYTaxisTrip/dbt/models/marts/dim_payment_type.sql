select
    distinct payment_id,
    case
	when payment_id = 0 then 'Flex Fare Trip'
	when payment_id = 1 then 'Credit Card'
	when payment_id = 2 then 'Cash'
	when payment_id = 3 then 'No Charge'
	when payment_id = 4 then 'Dispute'
	when payment_id = 5 then 'Unknown'
	when payment_id = 6 then 'Voided'
	--  else 'Error'
    end as payment_name
from {{ ref('stg_yellow_taxis') }}
order by payment_id
