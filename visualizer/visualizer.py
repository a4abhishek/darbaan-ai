from abc import abstractmethod

import cv2


class Visualizer:
    @abstractmethod
    def visualize(self, frame: cv2.typing.MatLike, label: str, confidence: float):
        pass


class VisualizerStrategy(Visualizer):
    def __init__(self, visualizer: Visualizer):
        self._visualizer = visualizer

    def visualize(self, frame: cv2.typing.MatLike, label: str, confidence: float):
        self._visualizer.visualize(frame, label, confidence)
