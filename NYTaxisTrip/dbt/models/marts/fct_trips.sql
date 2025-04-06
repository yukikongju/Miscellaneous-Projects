-- yellow_taxis
SELECT
    1 as taxi_type_id,
    vendor_id,
    pickup_datetime,
    dropoff_datetime,
    pu_location_id,
    do_location_id,
    rate_code_id,
    payment_id,
    passenger_count,
    trip_distance,
    fare_amount
    mta_tax,
    extra,
    tip_amount,
    improvement_surcharge,
    congestion_surcharge,
    total_amount
FROM {{ ref('stg_yellow_taxis') }}
UNION ALL
SELECT
    2 as taxi_type_id,
    vendor_id,
    pickup_datetime,
    dropoff_datetime,
    pu_location_id,
    do_location_id,
    rate_code_id,
    payment_id,
    passenger_count,
    trip_distance,
    fare_amount
    mta_tax,
    extra,
    tip_amount,
    improvement_surcharge,
    congestion_surcharge,
    total_amount
from {{ ref('stg_green_taxis') }}
