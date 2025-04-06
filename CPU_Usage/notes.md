## Open influxDB

```
1. Find container_id corresponding to influxdb:1.8
> docker ps 

2. Execute db

> docker exec -it 46f5bd3e4fad influx
> show databases;
> use metricsdb;
> show measurements;

3. 

curl -G http://localhost:8086/query \
  --data-urlencode "db=metricsdb" \
  --data-urlencode "q=SHOW MEASUREMENTS;"
```

## Setting up InfluxDB in grafana

- URL: `http://influxdb:8086`
- Database: `metricsdb`
- HTTP Method: `GET`
- Leave user and password blank because they were left undefined



