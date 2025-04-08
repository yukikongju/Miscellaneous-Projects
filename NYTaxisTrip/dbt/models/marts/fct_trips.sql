with yellow_taxis as (
    select
	1 as taxi_type_id,
	*
    FROM {{ ref('stg_yellow_taxis') }}
), green_taxis as (
    select
	2 as taxi_type_id,
	*
    from {{ ref('stg_green_taxis') }}
), taxis as (
    select
        taxi_type_id,
        vendor_id,
        pickup_datetime,
        pickup_date,
        pickup_time,
        dropoff_datetime,
        dropoff_date,
        dropoff_time,
        pu_location_id,
        do_location_id,
        rate_code_id,
        payment_id,
        passenger_count,
        trip_distance,
        fare_amount,
        mta_tax,
        extra,
        tip_amount,
        improvement_surcharge,
        congestion_surcharge,
        total_amount
    from yellow_taxis
    union all
    select
        taxi_type_id,
        vendor_id,
        pickup_datetime,
        pickup_date,
        pickup_time,
        dropoff_datetime,
        dropoff_date,
        dropoff_time,
        pu_location_id,
        do_location_id,
        rate_code_id,
        payment_id,
        passenger_count,
        trip_distance,
        fare_amount,
        mta_tax,
        extra,
        tip_amount,
        improvement_surcharge,
        congestion_surcharge,
        total_amount
    from green_taxis
)

select
    {{ dbt_utils.generate_surrogate_key([
        'taxi_type_id',
        'vendor_id',
        'pickup_date',
        'pickup_time',
        'dropoff_date',
        'dropoff_time',
        'pu_location_id',
        'do_location_id'
    ]) }} as trip_id,
    *
from taxis
