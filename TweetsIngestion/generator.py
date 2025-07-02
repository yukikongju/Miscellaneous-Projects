from datetime import datetime
from enum import Enum

import logging
import json
import random
import requests


class EventType(Enum):
    LIKE = 0
    COMMENT = 1
    POST = 2
    SHARE = 3


def generate_event() -> dict:
    NUM_USERS = 1000
    NUM_POSTS = 500

    user_id = random.randint(0, NUM_USERS)
    post_id = random.randint(0, NUM_POSTS)
    event_type = random.choice(list(EventType))

    if event_type == EventType.LIKE:
        event = {
            "user_id": user_id,
            "event_type": event_type.name.lower(),
            "event_timestamp": datetime.now(),
            "event_params": {"post_id": post_id, "content": "difuahsfudpas"},
        }
    elif event_type in [EventType.COMMENT, EventType.POST, EventType.SHARE]:
        event = {
            "user_id": user_id,
            "event_type": event_type.name.lower(),
            "event_timestamp": datetime.now(),
            "event_params": {
                "post_id": post_id,
            },
        }
    return event


def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    url = "http://localhost:5000/send_event"
    headers = {"Content-type": "application/json", "Accept": "text/plain"}
    while True:
        event = generate_event()
        requests.post(url=url, data=json.dumps(event, sort_keys=True, default=str), headers=headers)
        logger.info(f"{event['user_id']}: {event['event_type']}")


if __name__ == "__main__":
    main()
