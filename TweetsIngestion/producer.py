from flask import Flask, request, jsonify

#  from kafka import KafkaProducer
#  from kafka.errors import KafkaError
#  import msgpack
#  import json

BROKER_PORT = "localhost:9092"
TOPIC_NAME = "events"
app = Flask(__name__)
#  producer = KafkaProducer(bootstrap_servers=[BROKER_PORT],
#                  value_serializer=lambda v: json.dumps(v).encode('utf-8'))


@app.route("/send_event", methods=["POST"])
def send_event():
    data = request.json
    if not data:
        return jsonify({"error": "Invalid Input"}), 400

    try:
        event = {
            "user_id": data.get("user_id"),
            "event_type": data.get("event_type"),
            "event_timestamp": data.get("event_timestamp"),
            "event_params": data.get("event_params"),
        }
        print(event)
        #  producer.send(TOPIC_NAME, event)
    except Exception as e:
        raise ValueError(f"Couldn't process event. Exit with error: {e}")

    return jsonify({"status": "sent"}), 200


if __name__ == "__main__":
    FLASK_PORT = 5000
    app.run(port=FLASK_PORT)
