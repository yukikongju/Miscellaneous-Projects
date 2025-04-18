{{ config(materialized='table') }}

with yellow_taxis as (
    SELECT
	"VendorID" as vendor_id,
	"tpep_pickup_datetime" as pickup_datetime,
        cast("tpep_pickup_datetime" as date) as pickup_date,
        cast("tpep_pickup_datetime" as time) as pickup_time,
	"tpep_dropoff_datetime" as dropoff_datetime,
        cast("tpep_dropoff_datetime" as date) as dropoff_date,
        cast("tpep_dropoff_datetime" as time) as dropoff_time,
	"passenger_count",
	"trip_distance",
	"RatecodeID" as rate_code_id,
	"store_and_fwd_flag",
	"PULocationID" as pu_location_id,
	"DOLocationID" as do_location_id,
	"payment_type" as payment_id,
	"fare_amount",
	"extra",
	"mta_tax",
	"tip_amount",
	"tolls_amount",
	"improvement_surcharge",
	"total_amount",
	"congestion_surcharge",
	"Airport_fee" as airport_fee
	--  "cbd_congestion_fee"
    --  FROM public.yellow_taxi_trips
    FROM {{ source('src_posgres', 'yellow_taxi_trips') }}
    ORDER BY tpep_pickup_datetime DESC
)

select * from yellow_taxis
