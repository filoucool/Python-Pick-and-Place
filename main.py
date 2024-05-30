import time
import cv2
import numpy as np
import torch
from xarm.wrapper import XArmAPI
import pyrealsense2 as rs
from ultralytics import YOLO


class RealSenseCamera:
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(self.config)

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return None
        color_image = np.asanyarray(color_frame.get_data())
        return color_image

    def release(self):
        self.pipeline.stop()


class BinPicking:
    def __init__(self, model_path, arm_ip):
        self.model = self.load_model(model_path)
        self.yolo_model = YOLO("yolov8x.pt")  # Load the YOLOv8 model
        self.camera = RealSenseCamera()
        self.arm = XArmAPI(arm_ip)
        self.home_position = [275, 0, 400, -180, 0, -90]
        self.place_position = [540, 225, 245, -180, 0, -90]
        self.tolerance = 10  # Tolerance for centering in pixels
        self.move_increment = 10  # Incremental move in mm for centering

        self.connect_arm()
        self.move_to_home()

    def load_model(self, model_path):
        try:
            model = torch.load(model_path)
            model.eval()
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    def connect_arm(self):
        self.arm.connect()
        print("Connected to xArm")

    def move_to_home(self):
        print("Moving to home position")
        self.arm.set_position(*self.home_position, wait=True)

    def move_to_place_position(self):
        print("Moving to place position")
        self.arm.set_position(*self.place_position, wait=True)

    def capture_image(self):
        print("Capturing image")
        image = self.camera.get_frame()
        if image is None:
            print("Failed to capture image")
        return image

    def process_image(self, image):
        print("Processing image")
        objects = self.detect_objects(image)
        return self.select_most_centered_object(objects)

    def detect_objects(self, image):
        print("Detecting objects")
        # Use YOLOv8 for object detection
        results = self.yolo_model(image)
        detections = results[0].boxes  # Accessing the first image's detections

        objects = []
        for box in detections:
            x1, y1, x2, y2 = box.xyxy[0].numpy()
            conf = box.conf[0].numpy()
            cls = int(box.cls[0].numpy())

            print(
                f"Detected object - x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}, conf: {conf}, cls: {cls}")  # Debug print for each detected object

            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            z = 0  # Placeholder for depth
            objects.append({'coordinates': (x_center, y_center, z), 'confidence': conf, 'class': cls})

            # Draw bounding box
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            # Draw label and confidence
            label = f'Class {cls}: {conf:.2f}'
            cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the image using OpenCV
        cv2.imshow("Camera View", image)
        cv2.waitKey(1)  # Display the image for 1 ms

        return objects

    def select_most_centered_object(self, objects):
        print("Selecting most centered object")
        if not objects:
            return None

        # Select the object closest to the center of the image
        image_center = (320, 240)  # Assuming image size 640x480
        objects.sort(key=lambda obj: (obj['coordinates'][0] - image_center[0]) ** 2 + (
                    obj['coordinates'][1] - image_center[1]) ** 2)
        return objects[0]

    def convert_coordinates(self, x_img, y_img):
        # Adjust the scale factors and offsets to match your setup
        scale_factor_x = 0.5  # Example scale factor, adjust based on your setup
        scale_factor_y = 0.5  # Example scale factor, adjust based on your setup
        offset_x = 358  # Example offset, adjust based on your setup
        offset_y = 35  # Example offset, adjust based on your setup

        # Invert the X axis if needed
        x_robot = offset_x - (x_img - 320) * scale_factor_x
        y_robot = offset_y + (240 - y_img) * scale_factor_y

        print(f"Image coordinates: ({x_img}, {y_img}) -> Robot coordinates: ({x_robot}, {y_robot})")  # Debug statement

        return x_robot, y_robot

    def center_object_in_frame(self, target):
        image_center_x = 320  # Center of the camera image in pixels
        image_center_y = 240

        while target is not None:
            x_img, y_img, z = target['coordinates']
            x_offset = x_img - image_center_x
            y_offset = y_img - image_center_y

            print(f"Current object position: ({x_img}, {y_img}), offsets: ({x_offset}, {y_offset})")

            # Check if the object is within the tolerance range
            if abs(x_offset) <= self.tolerance and abs(y_offset) <= self.tolerance:
                print("Object is centered")
                break

            # Calculate the direction to move
            x_move = -self.move_increment if x_offset > 0 else self.move_increment
            y_move = -self.move_increment if y_offset > 0 else self.move_increment

            # Move the robot by the calculated increments to center the object
            self.arm.set_position(
                self.arm.position[0] + x_move,
                self.arm.position[1] + y_move,
                self.arm.position[2],  # Keep Z constant
                -180, 0, -90,
                wait=True
            )

            # Capture a new image and re-detect the object
            time.sleep(1)  # Allow some time for the robot to stabilize
            image = self.capture_image()
            target = self.process_image(image)

            if target is None:
                print("Failed to re-detect the object after centering attempt")
                break

    def pick_and_place(self):
        image = self.capture_image()
        if image is not None:
            target = self.process_image(image)

            if target:
                # Center the object in the camera frame
                self.center_object_in_frame(target)

                # Re-capture and process the image to get the updated coordinates
                image = self.capture_image()
                target = self.process_image(image)
                if target:
                    x_img, y_img, z = target['coordinates']
                    x_robot, y_robot = self.convert_coordinates(x_img, y_img)

                    # Apply safety checks
                    if x_robot <= 150:
                        print(f"Invalid X coordinate: {x_robot}. Skipping this target.")
                        return

                    x_robot = max(x_robot, 150)
                    y_robot = max(min(y_robot, 540), -540)

                    print(f"Moving to pick position: {x_robot}, {y_robot}, {225 - 278}")
                    self.move_to_position(x_robot, y_robot, 225 - 278)  # Move above the object
                    print(f"Moving down to pick: {x_robot}, {y_robot}, 225")
                    self.move_to_position(x_robot, y_robot, 225)  # Move down to pick the object
                    # self.grasp_object()  # Implement grasp function
                    print("Moving to place position")
                    self.move_to_place_position()
                    # self.release_object()  # Implement release function
                    self.move_to_home()
                else:
                    print("No target found after centering")
            else:
                print("No target found")
        else:
            print("No image captured")

    def move_to_position(self, x, y, z):
        print(f"Moving to position: {x}, {y}, 225")
        self.arm.set_position(x, y, 225, -180, 0, -90, wait=True)

    def run(self):
        try:
            while True:
                self.pick_and_place()
                time.sleep(1)  # Adjust as needed
        except KeyboardInterrupt:
            print("Bin picking process interrupted.")
        finally:
            self.camera.release()
            self.arm.disconnect()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    bin_picking = BinPicking(model_path='model.h5', arm_ip='192.168.1.219')
    bin_picking.run()
