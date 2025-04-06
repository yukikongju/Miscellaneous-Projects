select
    distinct vendor_id,
    case
        when vendor_id = 1 then 'Creative Mobile Technologies'
        when vendor_id = 2 then 'VeriFone Inc'
        when vendor_id = 6 then 'Myle Technologies'
        when vendor_id = 7 then 'Helix'
        --  else 'Unknown'
    end as vendor_name
from {{ ref('stg_yellow_taxis') }}
order by vendor_id
