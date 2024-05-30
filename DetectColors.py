import numpy as np
import cv2
import pyrealsense2 as rs
from xarm.wrapper import XArmAPI
import argparse
import time


class ColorObjectDetector:
    def __init__(self, arm_ip, camera_offset_x, camera_offset_y, camera_offset_z, tcp_offset_z, home_position):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.color_ranges = {
            'red': ((0, 120, 70), (10, 255, 255)),
            'green': ((35, 40, 40), (85, 255, 255)),
            'blue': ((100, 150, 0), (130, 255, 255)),
            'yellow': ((20, 100, 100), (30, 255, 255)),
            'pink': ((145, 100, 100), (170, 255, 255)),
            'orange': ((11, 100, 100), (25, 255, 255)),
            'cyan': ((85, 150, 100), (95, 255, 255))
        }
        self.arm = XArmAPI(arm_ip)
        self.arm.connect()

        # Offsets in mm
        self.camera_offset_x = camera_offset_x
        self.camera_offset_y = camera_offset_y
        self.camera_offset_z = camera_offset_z
        self.tcp_offset_z = tcp_offset_z
        self.home_position = home_position

    def start_camera(self):
        self.pipeline.start(self.config)

    def stop_camera(self):
        self.pipeline.stop()

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        return color_image, depth_image

    def detect_objects_by_color(self, image):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        detections = []

        for color_name, (lower, upper) in self.color_ranges.items():
            mask = cv2.inRange(hsv_image, lower, upper)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 500:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cx, cy = x + w // 2, y + h // 2
                    detections.append((cx, cy, w, h, color_name))

        return detections

    def calculate_position(self, cx, cy, distance, depth_intrinsics):
        point = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [cx, cy], distance)
        point = np.array(point)

        # Adjust for camera offset
        point[0] += self.camera_offset_x / 1000.0
        point[1] += self.camera_offset_y / 1000.0
        point[2] += (self.camera_offset_z + self.tcp_offset_z) / 1000.0

        return point

    def draw_detections(self, color_image, depth_image, depth_scale, detections):
        depth_intrinsics = self.pipeline.get_active_profile().get_stream(
            rs.stream.depth).as_video_stream_profile().get_intrinsics()

        for (cx, cy, w, h, color_name) in detections:
            color = (0, 0, 0)
            if color_name == 'red':
                color = (0, 0, 255)
            elif color_name == 'green':
                color = (0, 255, 0)
            elif color_name == 'blue':
                color = (255, 0, 0)
            elif color_name == 'yellow':
                color = (0, 255, 255)
            elif color_name == 'pink':
                color = (255, 105, 180)
            elif color_name == 'orange':
                color = (255, 165, 0)
            elif color_name == 'cyan':
                color = (255, 255, 0)

            x = cx - w // 2
            y = cy - h // 2
            cv2.rectangle(color_image, (x, y), (x + w, y + h), color, 2)

            # Get the distance from the depth image
            distance = depth_image[cy, cx] * depth_scale

            # Validate the distance measurement
            if distance > 0 and distance < 1.0:  # Assuming objects are within 1 meter
                # Calculate 3D coordinates
                point = self.calculate_position(cx, cy, distance, depth_intrinsics)

                cv2.putText(color_image, f'{color_name} ({point[0]:.2f},{point[1]:.2f},{point[2]:.2f})m', (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                mask = np.zeros_like(color_image, dtype=np.uint8)
                mask[y:y + h, x:x + w] = color
                color_image = cv2.addWeighted(color_image, 1.0, mask, 0.5, 0)

                # Capture image of the object being picked
                self.show_object_image(color_image, x, y, w, h)

                # Move the robot to the detected object's position
                self.move_robot_to_object(point)

        return color_image

    def show_object_image(self, image, x, y, w, h):
        object_image = image[y:y + h, x:x + w]
        cv2.imshow('Object to Pick', object_image)
        cv2.waitKey(500)  # Display the image for 500ms

    def move_robot_to_object(self, point):
        # Convert point coordinates to millimeters
        x, y, z = point * 1000
        # Ensure x position is always more than 150mm for safety
        if x < 150:
            print(f"Skipping movement to x={x} as it is less than 150mm for safety.")
            return
        # Ensure z position is always more than 150mm for safety
        if z < 150:
            z = 150
        # Move the robot to the position
        print(f"Moving to: x={x:.2f}, y={y:.2f}, z={z:.2f}")
        self.arm.set_position(x=x, y=y, z=z, wait=True)
        # Return to home position after pick-up
        self.return_to_home()

    def return_to_home(self):
        print("Returning to home position")
        self.arm.set_position(x=self.home_position[0], y=self.home_position[1], z=self.home_position[2], wait=True)
        # Add a delay to allow the camera to stabilize and provide accurate depth measurements
        time.sleep(2)

    def run(self):
        self.start_camera()
        try:
            # Move to home position at the start
            self.return_to_home()

            while True:
                # Get frames and detect objects
                color_frame, depth_frame = self.get_frame()
                depth_scale = self.pipeline.get_active_profile().get_device().first_depth_sensor().get_depth_scale()
                detections = self.detect_objects_by_color(color_frame)

                # Show all objects detected and masked
                output_frame = self.draw_detections(color_frame, depth_frame, depth_scale, detections)
                cv2.imshow('Color Detection', output_frame)
                cv2.waitKey(500)  # Display the image for 500ms

                if detections:
                    # If detections are found, move robot to object
                    self.draw_detections(color_frame, depth_frame, depth_scale, detections)

        finally:
            self.stop_camera()
            self.arm.disconnect()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Color Object Detection and Robot Control')
    parser.add_argument('--arm_ip', type=str, default='192.168.1.219', help='IP address of the xArm 6')
    parser.add_argument('--camera_offset_x', type=float, default=100, help='Camera X offset in mm')
    parser.add_argument('--camera_offset_y', type=float, default=0, help='Camera Y offset in mm')
    parser.add_argument('--camera_offset_z', type=float, default=30, help='Camera Z offset in mm')
    parser.add_argument('--tcp_offset_z', type=float, default=0, help='Gripper TCP offset in mm')
    parser.add_argument('--home_position', type=float, nargs=3, default=[250, 0, 350],
                        help='Home position [X, Y, Z] in mm')
    args = parser.parse_args()

    detector = ColorObjectDetector(args.arm_ip, args.camera_offset_x, args.camera_offset_y, args.camera_offset_z,
                                   args.tcp_offset_z, args.home_position)
    detector.run()
