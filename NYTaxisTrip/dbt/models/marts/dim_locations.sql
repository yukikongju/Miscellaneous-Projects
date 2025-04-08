select
    {{ dbt_utils.generate_surrogate_key(['location_id']) }} as location_key,
    location_id,
    borough,
    zone,
    service_zone
from {{ ref('locations') }}
