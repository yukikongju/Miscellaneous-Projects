# Tweets Ingestion

Simulation to ingest tweets for learning purposes.

Events:
- posts
- like
- comment
- share

Concept Applied:
- cumulative table
- Kafka
- Spark

## How to run

In separate terminal:

```{sh}
docker compose up

python3 producer.py

python3 consumer.py

python3 generator.py
```
