# How Tos

TODOs
1. [ ] Docker
2. [ ] Airflow => find dags inside webserver

## 1. Postgres and pgAdmin4 setup with Docker for data ingestion

### Initialize postgres database and ingest 1 parquet file

```{zsh}
1. docker compose up
2.
URL="https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet"
python3 ingest_data.py \
  --user=root \
  --password=root \
  --host=localhost \
  --port=5432 \
  --db=ny_taxi \
  --table_name=yellow_taxi_trips \
  --url=${URL}

```

### How to open postgres database in the terminal

```{zsh}
pgcli -h localhost -p 5432 -u root -d ny_taxi

> \dt
> \d yellow_taxi_trips

```

### How to query inside pgAdmin4

pgAdmin4 is a wrapper GUI around postgres. To query inside pgAdmin4, we need
to create that server. However, because the postgres database and the gui
resides on different docker containers, we need to link them using
`pg-network`


```
1. Create network to link containers

> docker network create pg-network

2. Reopen postgres connection and pgadmin4 connection in separate terminal window

> window 1:
docker run -it \
  -e POSTGRES_USER="root" \
  -e POSTGRES_PASSWORD="root" \
  -e POSTGRES_DB="ny_taxi" \
  -v $(pwd)/ny_taxi_postgres_data:/var/lib/postgresql/data \
  -p 5432:5432 \
  --network=pg-network \
  --name pg-database \
  postgres:13

> window 2:
docker run -it \
  -e PGADMIN_DEFAULT_EMAIL="admin@admin.com" \
  -e PGADMIN_DEFAULT_PASSWORD="root" \
  -p 8080:80 \
  --network=pg-network \
  --name pgadmin \
  dpage/pgadmin4

```

We can check if the containers are on the same network using
`docker network inspect pg-network`. We should be able to see `pgadmin` and
`pg-database` under "Containers". To manually connect them:

```
docker network connect <network_name> pgadmin
docker network connect <network_name> pgdatabase
```

We are now ready to create a server!

To do so,
1. "Servers" > "Register" > "Server"
2. In "General", add custom name. I chose "local"
3. In "Connections", use the following credentials, then click "Save"
    * hostname: `pg-database`
    * port: `5432`
    * maintenance db: `postgres`
    * username: `admin`
    * password: `root`

The table `yellow_taxi_trips` can be found under "local" > "ny_taxi" > "Schemas" > "Tables"

Notes:
To do the network automatically, we can add the following in
our docker-compose.yaml file

```
networks:
  default:
    name: airflow-network
```


### How to dockerize the ingestion script (TODO)


## 2. Setting up Airflow

[airflow docker tutorial](https://medium.com/@le.oasis/apache-airflow-docker-tutorial-2c8c152a074a)

```

mkdir -p airflow/dags airflow/logs airflow/plugins


# verifying if dags are recognized
docker exec -it airflow-webserver ls /opt/airflow/dags

# spinning up single docker container
docker compose up -d <CONTAINER_NAME>
```
