postgres:
	pgcli -h localhost -p 5432 -u root -d ny_taxi

docker_prune:
	docker system prune -a --volumes -f

airflow_prep:
	mkdir -p airflow/dags airflow/logs airflow/plugins

dbt:
	mkdir -p dbt/models dbt/analyses dbt/tests dbt/seeds dbt/macros dbt/snapshots dbt/logs dbt/seeds
	dbt init taxis
	dbt compile
	dbt debug

download_high_volume_data:
	wget https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_2024-01.parquet
