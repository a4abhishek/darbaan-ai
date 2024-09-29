import concurrent.futures


class FaceRecognitionState:
    def __init__(self, frame, label, confidence):
        self.Frame = frame
        self.Label = label
        self.Confidence = confidence


class FaceRecognitionSubject:
    def __init__(self, state: FaceRecognitionState):
        self._state = state
        self._executor = concurrent.futures.ThreadPoolExecutor()
        self._observers = []

    def add_observer(self, observer):
        self._observers.append(observer)

    def remove_observer(self, observer):
        self._observers.remove(observer)

    def _notify_observers(self):
        for observer in self._observers:
            self._executor.submit(observer.update, self._state)

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, new_state):
        self._state = new_state
        self._notify_observers()
