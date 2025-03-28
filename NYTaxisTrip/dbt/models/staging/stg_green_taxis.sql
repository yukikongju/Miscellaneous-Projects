WITH green_taxis AS (
    SELECT
        "VendorID" AS vendor_id,
        "lpep_pickup_datetime",
        "lpep_dropoff_datetime",
        "store_and_fwd_flag",
        "RatecodeID" AS rate_code_id,
        "PULocationID" AS pu_location_id,
        "DOLocationID" AS do_location_id,
        "passenger_count",
        "trip_distance",
        "fare_amount",
        "extra",
        "mta_tax",
        "tip_amount",
        "tolls_amount",
        "improvement_surcharge",
        "total_amount",
        "payment_type",
        "trip_type",
        "congestion_surcharge"
    FROM public.green_taxi_trips
    ORDER BY lpep_pickup_datetime DESC
)

SELECT *
FROM green_taxis;
