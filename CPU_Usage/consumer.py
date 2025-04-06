from kafka import KafkaConsumer
from influxdb import InfluxDBClient
import json

#  import time

def json_deserializer(data):
    return json.loads(data.decode('utf-8'))


# Kafka configuration
KAFKA_BOOTSTRAP_SERVERS = ["localhost:9092"]
#  KAFKA_BOOTSTRAP_SERVERS = ["kafka:9092"]
TOPIC_NAME = "cpu-metrics"

# Create Kafka Consumer
consumer = KafkaConsumer(
    TOPIC_NAME,
    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
    value_deserializer=json_deserializer,
    auto_offset_reset="earliest",
    enable_auto_commit=True,
    group_id="cpu-metrics-group",
)

# Connect to InfluxDB
influx = InfluxDBClient(host="localhost", port=8086)
influx.switch_database("metricsdb")

# Consume and write to InfluxDB
try:
    for message in consumer:
        data = message.value
        point = {
            "measurement": "cpu",
            "tags": {"host": data["host"]},
            "time": data["timestamp"],
            "fields": {
                "cpu_usage": float(data["cpu_usage"]), 
                "batch": int(data["batch"])},
        }
        influx.write_points([point], time_precision='s')
        print(f"Wrote to InfluxDB: {point}")

except Exception as e:
    print(f"Error while consuming or writing: {e}")

finally:
    consumer.close()
    influx.close()
