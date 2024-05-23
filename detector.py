import cv2
import pyrealsense2 as rs
import numpy as np
import logging
import json
import time

class RealSenseCubeDetector:
    def __init__(self, min_area=100, resolution=(640, 480), output_file=None):
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

        self.color_ranges = self.define_color_ranges()
        self.min_area = min_area  # Minimum area of the contour to be considered as a cube
        self.kernel = np.ones((5, 5), np.uint8)  # Morphological kernel
        self.output_file = output_file

    def define_color_ranges(self):
        return {
            'red': (np.array([0, 120, 70]), np.array([10, 255, 255])),
            'green': (np.array([40, 40, 40]), np.array([80, 255, 255])),
            'blue': (np.array([100, 150, 150]), np.array([140, 255, 255])),
            'yellow': (np.array([20, 100, 100]), np.array([30, 255, 255])),
            'cyan': (np.array([80, 100, 100]), np.array([100, 255, 255])),
            'orange': (np.array([10, 100, 100]), np.array([25, 255, 255])),
            'pink': (np.array([145, 60, 65]), np.array([165, 255, 255]))
        }

    def process_frame(self, color_image, depth_frame):
        detected_cubes = []
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

        for color, (lower, upper) in self.color_ranges.items():
            mask = cv2.inRange(hsv, lower, upper)
            mask = cv2.dilate(mask, self.kernel, iterations=1)
            mask = cv2.erode(mask, self.kernel, iterations=1)
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) > self.min_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    center = (x + w // 2, y + h // 2)
                    depth = depth_frame.get_distance(center[0], center[1])
                    detected_cubes.append({
                        'color': color,
                        'position': center,
                        'rectangle': (x, y, w, h),
                        'depth': depth
                    })
        return detected_cubes

    def detect_cubes(self):
        if not self.pipeline:
            return []

        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not color_frame or not depth_frame:
            return []

        color_image = np.asanyarray(color_frame.get_data())
        detected_cubes = self.process_frame(color_image, depth_frame)

        for cube in detected_cubes:
            x, y, w, h = cube['rectangle']
            cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = f"{cube['color']} ({cube['depth']:.2f}m)"
            cv2.putText(color_image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow('RealSense Cube Detection', color_image)
        cv2.waitKey(1)  # Ensure the window refreshes to show the detection

        logging.info(f"Detected cubes: {detected_cubes}")

        if self.output_file:
            with open(self.output_file, 'w') as f:
                json.dump(detected_cubes, f)

        return detected_cubes

    def stop(self):
        if self.pipeline:
            self.pipeline.stop()
        cv2.destroyAllWindows()

# External callable function
def run_realsense_cube_detector(min_area=100, width=640, height=480, log_level="INFO", output_file=None):
    logging.basicConfig(level=log_level.upper())
    detector = RealSenseCubeDetector(min_area=min_area, resolution=(width, height), output_file=output_file)
    detected_cubes = detector.detect_cubes()
    detector.stop()
    return detected_cubes
