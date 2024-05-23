import cv2
import pyrealsense2 as rs
import numpy as np
import logging
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


class RealSenseObjectDetector:
    def __init__(self, model_path, label_path, resolution=(640, 480), output_file=None):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, resolution[0], resolution[1], rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, resolution[0], resolution[1], rs.format.bgr8, 30)
        try:
            self.pipeline.start(self.config)
        except Exception as e:
            logging.error(f"Failed to start RealSense pipeline: {e}")
            self.pipeline = None

        # Allow the camera to warm up
        time.sleep(2)

        # Load the model
        self.model = load_model(model_path)

        # Load labels
        with open(label_path, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]

        self.output_file = output_file

    def detect_objects(self, image):
        height, width = image.shape[:2]
        input_image = cv2.resize(image, (224, 224))  # Resize to the model's input size
        input_image = preprocess_input(input_image)  # Preprocess the image
        input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension

        predictions = self.model.predict(input_image)
        detected_objects = self.postprocess_predictions(predictions, width, height)
        return detected_objects

    def postprocess_predictions(self, predictions, image_width, image_height):
        detected_objects = []
        # This will vary depending on your model's output format
        for prediction in predictions:
            confidence = prediction[4]  # Assuming confidence score is at index 4
            if confidence > 0.5:  # Confidence threshold
                class_id = int(prediction[5])  # Assuming class id is at index 5
                box = prediction[:4] * np.array([image_width, image_height, image_width, image_height])
                (x, y, x1, y1) = box.astype("int")
                label = self.labels[class_id] if class_id < len(self.labels) else "Unknown"
                detected_objects.append({
                    'label': label,
                    'confidence': confidence,
                    'rectangle': (x, y, x1 - x, y1 - y)
                })
        return detected_objects

    def detect_and_display(self):
        if not self.pipeline:
            return []

        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not color_frame or not depth_frame:
            return []

        color_image = np.asanyarray(color_frame.get_data())
        detected_objects = self.detect_objects(color_image)

        for obj in detected_objects:
            x, y, w, h = obj['rectangle']
            label = obj['label']
            confidence = obj['confidence']
            cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = f"{label}: {confidence:.2f}"
            cv2.putText(color_image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow('RealSense Object Detection', color_image)
        cv2.waitKey(1)  # Ensure the window refreshes to show the detection

        logging.info(f"Detected objects: {detected_objects}")

        if self.output_file:
            with open(self.output_file, 'w') as f:
                json.dump(detected_objects, f)

        return detected_objects

    def stop(self):
        if self.pipeline:
            self.pipeline.stop()
        cv2.destroyAllWindows()


# External callable function
def run_realsense_object_detector(model_path, label_path, resolution=(640, 480), log_level="INFO", output_file=None):
    logging.basicConfig(level=log_level.upper())
    detector = RealSenseObjectDetector(model_path=model_path, label_path=label_path, resolution=resolution,
                                       output_file=output_file)
    detected_objects = detector.detect_and_display()
    detector.stop()
    return detected_objects

# Example of how to use in another script
# from realsense_cube_detector import run_realsense_object_detector
# detected_objects = run_realsense_object_detector(model_path="model.h5", label_path="labels.txt", resolution=(640, 480), log_level="INFO", output_file="output.json")
# print(detected_objects)
