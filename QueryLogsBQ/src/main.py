import json
import os
import requests
import ssl

from datetime import datetime

#  from dotenv import load_dotenv
from flask import Flask, request
from SlackNotifier import SlackNotifier

app = Flask(__name__)
ssl._create_default_https_context = ssl._create_stdlib_context
#  load_dotenv()
SLACK_TOKEN = os.getenv("SLACK_TOKEN")
SLACK_CHANNEL_NAME = os.getenv("SLACK_CHANNEL_NAME")


@app.route("/message", methods=["POST"])
def send_message():
    envelope = request.get_json()
    if not envelope:
        return "Bad Request", 400

    # pub/sub wraps message in {"message": {"data": "..."}}
    msg = envelope.get("message", {})
    data = msg.get("data")
    if data:  # send message to slack
        notifier = SlackNotifier(slack_token=SLACK_TOKEN)
        message = f"testing: {data}"
        notifier.send_message(channel=CHANNEL_NAME, message=message)
        print(f"Message sent in slack at {datetime.now()}")

    return "OK", 200


if __name__ == "__main__":
    app.run(port=8080)
