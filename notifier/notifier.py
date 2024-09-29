from abc import abstractmethod


class Notifier:
    @abstractmethod
    def send_notification(self, message: str):
        pass


class NotificationStrategy(Notifier):
    def __init__(self, notifier: Notifier):
        self._notifier = notifier

    def send_notification(self, message: str):
        self._notifier.send_notification(message)
