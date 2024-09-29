import pickle
import time

import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from openvino.inference_engine import IECore

from face_recognizer.subject import FaceRecognitionSubject, FaceRecognitionState
from visualizer.visualizer import Visualizer


class FaceRecognizer:
    def __init__(self, svm_classifier_path, label_encoder_path, visualizer: Visualizer, subject: FaceRecognitionSubject):
        self._svm_classifier_path = svm_classifier_path
        self._label_encoder_path = label_encoder_path
        self._visualizer = visualizer
        self._subject = subject

    @staticmethod
    def preprocess_face(face):
        face = cv2.resize(face, (160, 160))
        face = face.astype('float32') / 255.0
        face = np.transpose(face, (2, 0, 1))  # HWC to CHW
        face = np.expand_dims(face, axis=0)  # Add batch dimension
        return face

    def _load_models(self):
        # Load the trained SVM model and label encoder
        svm_classifier = pickle.load(open(self._svm_classifier_path, 'rb'))
        label_encoder = pickle.load(open(self._label_encoder_path, 'rb'))

        # Check if CUDA is available and set device accordingly
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load FaceNet model
        facenet_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

        # Initialize MTCNN for face detection
        detector = MTCNN(keep_all=False, device=device)

        # Set up OpenVINO Inference Engine if needed
        use_openvino = False
        if not torch.cuda.is_available():
            try:
                ie = IECore()
                net = ie.read_network(model="facenet_model.xml", weights="facenet_model.bin")
                exec_net = ie.load_network(network=net, device_name="GPU")
                input_blob = next(iter(net.input_info))
                output_blob = next(iter(net.outputs))
                use_openvino = True
            except Exception as e:
                print("OpenVINO initialization failed:", e)
                exec_net, input_blob, output_blob = None, None, None
        else:
            exec_net, input_blob, output_blob = None, None, None

        return svm_classifier, label_encoder, facenet_model, detector, device, use_openvino, exec_net, input_blob, output_blob

    def extract_face_embeddings(self, image_data, facenet_model, detector, device, use_openvino, exec_net, input_blob, output_blob):
        img_rgb = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        faces, probs = detector.detect(img_rgb)
        if faces is not None and probs[0] > 0.90:
            x1, y1, x2, y2 = faces[0]
            face = img_rgb[int(y1):int(y2), int(x1):int(x2)]
            if face.size == 0:
                return None
            face = self.preprocess_face(face)

            if use_openvino:
                exec_net.start_async(request_id=0, inputs={input_blob: face})
                if exec_net.requests[0].wait(-1) == 0:
                    embedding = exec_net.requests[0].outputs[output_blob]
                return embedding[0]
            else:
                face = torch.tensor(face).to(device)
                with torch.no_grad():
                    embedding = facenet_model(face)
                return embedding[0].cpu().numpy()
        else:
            return None

    def recognize_faces(self):
        svm_classifier, label_encoder, facenet_model, detector, device, use_openvino, exec_net, input_blob, output_blob = self._load_models()

        # Open the camera
        cap = cv2.VideoCapture(0)

        prev_identity = ''
        time_identify = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            embedding = self.extract_face_embeddings(frame, facenet_model, detector, device, use_openvino, exec_net, input_blob, output_blob)
            if embedding is not None:
                embedding = embedding.reshape(1, -1)
                prediction = svm_classifier.predict(embedding)
                prob = svm_classifier.predict_proba(embedding)

                label = label_encoder.inverse_transform(prediction)[0]
                confidence = np.max(prob)

                self._visualizer.visualize(frame, label, confidence)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                if time.time() - time_identify > 60:  # 60 seconds have elapsed
                    prev_identity = ''
                    time_identify = time.time()
                if label != prev_identity and confidence > 0.8:
                    prev_identity = label
                    time_identify = time.time()

                    self._subject.state = FaceRecognitionState(frame, label, confidence)

        cap.release()
        cv2.destroyAllWindows()
