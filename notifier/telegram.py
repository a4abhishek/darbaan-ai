import requests

from notifier.notifier import Notifier


class TelegramNotifier(Notifier):
    def __init__(self, bot_token: str, chat_id: str):
        self._bot_token = bot_token
        self._chat_id = chat_id
        self._url = f'https://api.telegram.org/bot{self._bot_token}/sendMessage'

    def send_notification(self, message: str):
        payload = {'chat_id': self._chat_id, 'text': message}
        requests.post(self._url, data=payload)
