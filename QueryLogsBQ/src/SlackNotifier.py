import logging
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError


class SlackNotifier:

    def __init__(self, slack_token: str):
        self.slack_token = slack_token
        self.client = self.__init_client()

    def __init_client(self):
        """
        Initialize slack client
        """
        try:
            client = WebClient(token=self.slack_token)
            return client
        except:
            raise ValueError(
                f"Slack Web Client connection was unsuccessful. Please check token or connection."
            )

    def send_message(self, channel: str, message: str):
        """
        Parameters
        ----------
        channel:
            > channel where message should be sent to
        message: str
            > message to be sent in slack channel

        Docs:
        - https://www.datacamp.com/tutorial/how-to-send-slack-messages-with-python
        - https://api.slack.com/apps/A06JUC9PWEA?created=1

        Example
        -------
        > slack_notifier = SlackNotifier(slack_token=SLACK_TOKEN)
        > slack_notifier.send_message(channel="tmp", message="testing")
        """
        try:
            results = self.client.chat_postMessage(
                channel=channel, text=message, username="DS-Bot"
            )
            logging.log(
                0,
                f"Warnings message was sent to slack in channel {channel} with message '{message}'",
            )
        except SlackApiError as e:
            logging.error(e)
            logging.error(
                "Slack message was not sent due to an error. Make sure that your app is added in the channel via `integration`"
            )
