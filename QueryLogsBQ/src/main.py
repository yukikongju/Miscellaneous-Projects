import os
import ssl

from dotenv import load_dotenv
from SlackNotifier import SlackNotifier


def main():
    ssl._create_default_https_context = ssl._create_stdlib_context
    load_dotenv()
    SLACK_TOKEN = os.getenv("SLACK_TOKEN")
    CHANNEL_NAME = "tmp"

    notifier = SlackNotifier(slack_token=SLACK_TOKEN)
    notifier.send_message(channel=CHANNEL_NAME, message="testing")


if __name__ == "__main__":
    main()
