from kafka import KafkaConsumer
import json


BROKER_PORT = "localhost:9092"
TOPIC_NAME = "events"
consumer = KafkaConsumer(
    TOPIC_NAME,
    bootstrap_servers=BROKER_PORT,
    value_deserializer=lambda m: json.loads(m.decode("utf-8")),
)

#  consumer.subscribe()
for message in consumer:
    print(f"{message.value}")
