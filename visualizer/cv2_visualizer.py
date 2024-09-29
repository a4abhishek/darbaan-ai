import cv2

from visualizer.visualizer import Visualizer


class CV2Visualizer(Visualizer):
    def visualize(self, frame: cv2.typing.MatLike, label: str, confidence: float):
        cv2.putText(frame, f"{label} ({confidence:.2f})", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Face Recognition', frame)
