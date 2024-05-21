# Commit message: Add command-line arguments for min area and stream resolution

import cv2
import pyrealsense2 as rs
import numpy as np
import argparse

class RealSenseCubeDetector:
    def __init__(self, min_area=100, resolution=(640, 480)):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, resolution[0], resolution[1], rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, resolution[0], resolution[1], rs.format.bgr8, 30)
        try:
            self.pipeline.start(self.config)
        except Exception as e:
            print(f"Failed to start RealSense pipeline: {e}")
            self.pipeline = None

        self.color_ranges = self.define_color_ranges()
        self.min_area = min_area  # Minimum area of the contour to be considered as a cube
        self.kernel = np.ones((5, 5), np.uint8)  # Morphological kernel

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

    def process_frame(self, color_image):
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
                    detected_cubes.append({
                        'color': color,
                        'position': center
                    })
        return detected_cubes

    def detect_cubes(self):
        if not self.pipeline:
            return []

        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return []

        color_image = np.asanyarray(color_frame.get_data())
        detected_cubes = self.process_frame(color_image)
        print(detected_cubes)
        return detected_cubes

    def stop(self):
        if self.pipeline:
            self.pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RealSense Cube Detector")
    parser.add_argument("--min_area", type=int, default=100, help="Minimum contour area to consider as a cube")
    parser.add_argument("--width", type=int, default=640, help="Width of the video stream")
    parser.add_argument("--height", type=int, default=480, help="Height of the video stream")
    args = parser.parse_args()

    detector = RealSenseCubeDetector(min_area=args.min_area, resolution=(args.width, args.height))
    try:
        while True:
            detected_cubes = detector.detect_cubes()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        detector.stop()
