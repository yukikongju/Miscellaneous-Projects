WITH green_taxis AS (
    SELECT
        "VendorID" AS vendor_id,
        "lpep_pickup_datetime" as pickup_datetime,
        cast("lpep_pickup_datetime" as date) as pickup_date,
        cast("lpep_pickup_datetime" as time) as pickup_time,
        "lpep_dropoff_datetime" as dropoff_datetime,
        cast("lpep_dropoff_datetime" as date) as dropoff_date,
        cast("lpep_dropoff_datetime" as time) as dropoff_time,
        "store_and_fwd_flag",
        "RatecodeID" AS rate_code_id,
        "PULocationID" AS pu_location_id,
        "DOLocationID" AS do_location_id,
        "payment_type" as payment_id,
        "passenger_count",
        "trip_distance",
        "fare_amount",
        "extra",
        "mta_tax",
        "tip_amount",
        "tolls_amount",
        "improvement_surcharge",
        "total_amount",
        "trip_type",
        "congestion_surcharge"
    --  FROM public.green_taxi_trips
    FROM {{ source('src_posgres', 'green_taxi_trips') }}
    ORDER BY lpep_pickup_datetime DESC
)

SELECT *
FROM green_taxis
