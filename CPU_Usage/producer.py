from kafka import KafkaProducer
import json
import time
import random


# Serialize Python dict to JSON bytes
def json_serializer(data):
    return json.dumps(data).encode("utf-8")


# Kafka configuration
KAFKA_BOOTSTRAP_SERVERS = ["localhost:9092"]
#  KAFKA_BOOTSTRAP_SERVERS = ["kafka:9092"]
TOPIC_NAME = "cpu-metrics"

# Create producer
producer = KafkaProducer(
    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
    value_serializer=json_serializer,
    acks="all",  # Ensure message is committed
    linger_ms=10,  # Slight batching to improve throughput
)

# Track duration
t0 = time.time()

# Produce messages
try:
    for i in range(1000):
        data = {
            "host": "machine-1",
            "batch": i,
            "cpu_usage": round(random.uniform(20.0, 80.0), 2),
            "timestamp": int(time.time()),
        }
        print(f"Producing: {data}")
        producer.send(TOPIC_NAME, value=data)
        time.sleep(0.5)

    producer.flush()
    print("All messages flushed.")

except Exception as e:
    print(f"Error during Kafka production: {e}")

finally:
    producer.close()
    t1 = time.time()
    print(f"Took {(t1 - t0):.2f} seconds")
