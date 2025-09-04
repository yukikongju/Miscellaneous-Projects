docs:
	dbt docs generate
	dbt docs serve --port 9000

create_tables:
	python3 -m src.tables.table_googleads
	python3 -m src.tables.table_facebookads
	python3 -m src.tables.table_appsflyergeo
