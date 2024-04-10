import imaplib
#  import email


class GmailManager:

    def __init__(self, username: str, password: str):
        self.email = username
        self.password = password
        self.imap_url = 'imap.gmail.com'

        self.mail = imaplib.IMAP4_SSL(self.imap_url)
        self.login()

    def login(self) -> bool:
        """
        User Login using Imap and gmail app password

        Returns
        -------
        success: bool
            > returns true if user successfully logged in; else false
        """
        res = self.mail.login(user=self.email, password=self.password)
        if res[0] == 'OK':
            print(f"Successfully logged into {self.email}!")
            return True

        print(f"Failed to logged into {self.email}")
        return False

    def get_raw_email_from_sender(self, sender: str):
        """
        Get all emails received from a selected sender

        Parameters
        ----------
        sender: str
            > email of the sender. Ex: 'example@gmail.com'

        Returns
        -------

        """
        # search for emails from sender in inbox
        self.mail.select('inbox')
        _, data = self.mail.search(None, 'FROM', sender)
        emails_ids = data[0].split()

        # get raw emails
        raw_emails = []
        for email_id in emails_ids:
            _, data = self.mail.fetch(email_id, '(RFC822)')
            raw_email = data[0][1]
            #  msg = email.message_from_bytes(raw_email)
            raw_emails.append(raw_email)
        return raw_emails
