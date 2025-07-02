from flask import Flask, request, jsonify

from kafka import KafkaProducer
from kafka.errors import KafkaError

#  import msgpack
import json
import logging

BROKER_PORT = "localhost:9092"
TOPIC_NAME = "events"
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_producer():
    try:
        producer = KafkaProducer(
            bootstrap_servers=[BROKER_PORT],
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            acks="all",
            retries=3,
        )
        logger.info("Kafka producer created succesfully")
        return producer
    except Exception as e:
        logger.error(f"Failed to create kafka producer: {e}")
        return None


producer = create_producer()


@app.route("/send_event", methods=["POST"])
def send_event():
    if not producer:
        return jsonify({"error": "Kafka producer not available"}), 500

    data = request.json
    if not data:
        return jsonify({"error": "Invalid Input"}), 400

    required_fields = ["user_id", "event_type", "event_timestamp", "event_params"]
    for field in required_fields:
        if not data.get(field):
            return jsonify({"error": "Missing required field '{field}'"})

    try:
        event = {
            "user_id": data.get("user_id"),
            "event_type": data.get("event_type"),
            "event_timestamp": data.get("event_timestamp"),
            "event_params": data.get("event_params"),
        }
        future = producer.send(TOPIC_NAME, event)
        producer.flush()
        return jsonify({"status": "sent", "event": event}), 200
    except KafkaError as e:
        logger.error(f"Kafka error: {e}")
        return jsonify({"error": f"Kafka error: {e}"}), 500
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return jsonify({"error": "Unexpected error occured when processing event: {str(e)}"}), 500


if __name__ == "__main__":
    FLASK_PORT = 5000
    app.run(port=FLASK_PORT)
