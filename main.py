from config import config
from face_recognizer.facenet import FaceRecognizer
from face_recognizer.subject import FaceRecognitionState, FaceRecognitionSubject
from notifier.notifier import NotificationStrategy, Notifier
from notifier.telegram import TelegramNotifier
from speaker.gttsspeaker import GTTSSpeaker
from speaker.speaker import Speaker, SpeakingStrategy
from visualizer.cv2_visualizer import CV2Visualizer
from visualizer.visualizer import VisualizerStrategy


class NotificationObserver:
    def __init__(self, notifier: Notifier):
        self._notifier = notifier

    def update(self, state: FaceRecognitionState):
        print(f"{state.Label} has arrived at the doorstep, sending notification...")
        notification_message = f"Welcome alert: {state.Label} has arrived at the doorstep!"
        self._notifier.send_notification(notification_message)


class SpeakerObserver:
    def __init__(self, speaker: Speaker):
        self._speaker = speaker

    def update(self, state: FaceRecognitionState):
        print(f"greeting {state.Label}...")
        greetings = f"Welcome {state.Label}, I have informed the household that you are at the doorstep."
        self._speaker.speak(greetings, 'en')


if __name__ == "__main__":
    # Create Notifier
    telegram_bot_token = config['telegram']['bot_token']
    telegram_chat_id = config['telegram']['chat_id']
    telegram_notifier = TelegramNotifier(telegram_bot_token, telegram_chat_id)
    notification_strategy = NotificationStrategy(telegram_notifier)

    # Create Speaker
    gtts_speaker = GTTSSpeaker()
    speaker_strategy = SpeakingStrategy(gtts_speaker)

    # Create the Face Subject and set Subscribers
    subject = FaceRecognitionSubject(FaceRecognitionState(None, '', ''))

    notification_observer = NotificationObserver(notification_strategy)
    speaker_observer = SpeakerObserver(speaker_strategy)

    subject.add_observer(notification_observer)
    subject.add_observer(speaker_observer)

    # Crate Visualizer
    cv2_visualizer = CV2Visualizer()
    visualizer_strategy = VisualizerStrategy(cv2_visualizer)

    # Create Face Recognizer
    svm_classifier_path = config["model"]["svm_classifier_path"]
    label_encoder_path = config["model"]["label_encoder_path"]
    face_recognizer = FaceRecognizer(svm_classifier_path, label_encoder_path, visualizer_strategy, subject)

    # Start service
    face_recognizer.recognize_faces()
