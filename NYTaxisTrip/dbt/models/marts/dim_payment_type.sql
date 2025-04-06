select
    {{ dbt_utils.generate_surrogate_key(['payment_id']) }} as payment_key,
    payment_id,
    payment_name
from {{ ref('payment_types') }}
