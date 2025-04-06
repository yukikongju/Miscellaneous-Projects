select
    {{ dbt_utils.generate_surrogate_key(['taxi_type_id']) }} as taxi_type_key,
    taxi_type_id,
    taxi_type_name
from {{ ref('taxi_types') }}
