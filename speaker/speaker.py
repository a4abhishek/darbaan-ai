from abc import abstractmethod


class Speaker:
    @abstractmethod
    def speak(self, message: str, lang: str):
        pass


class SpeakingStrategy(Speaker):
    def __init__(self, speaker: Speaker):
        self._speaker = speaker

    def speak(self, message: str, lang: str = 'en'):
        self._speaker.speak(message, lang)
