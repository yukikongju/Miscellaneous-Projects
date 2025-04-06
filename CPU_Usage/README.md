# CPU Usage

Monitoring CPU usage and other metrics

Metrics:
- CPU Usage
- Keys Clicked

## Technologies

- Data Ingestion: Kafka
- Database: InfluxDB
- Visualization: Grafana

## Architecture Overview

Python Producer --> Kafka Topic --> Python Consumer --> InfluxDB --> Grafana


Python App ──▶ Logs ──▶ Loki
          └─▶ Metrics ──▶ Mimir (Prometheus)
          └─▶ Traces ──▶ Tempo
                      ↓
                 Visualized in Grafana

## Requirements

```
brew install grafana influxdb
```

- `influxd`


## Documentation

- [Grafana - Getting Started](https://grafana.com/tutorials/grafana-fundamentals/?utm_source=grafana_gettingstarted#introduction)
