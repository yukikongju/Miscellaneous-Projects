{{ config(materialized='table') }}

with yellow_taxis as (
    SELECT
	"VendorID",
	"tpep_pickup_datetime",
	"tpep_dropoff_datetime",
	"passenger_count",
	"trip_distance",
	"fare_amount",
	"total_amount"
    FROM
	public.yellow_taxi_trips
)

select * from yellow_taxis
