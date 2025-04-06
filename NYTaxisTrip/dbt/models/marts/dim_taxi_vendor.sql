select
    {{ dbt_utils.generate_surrogate_key(['vendor_id']) }} as payment_key,
    vendor_id,
    vendor_name
from {{ ref('vendors') }}
