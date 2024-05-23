# main.py

import argparse
import logging
from detector import run_realsense_cube_detector

def main():
    parser = argparse.ArgumentParser(description="Run RealSense Cube Detector")
    parser.add_argument("--min_area", type=int, default=600, help="Minimum contour area to consider as a cube")
    parser.add_argument("--width", type=int, default=640, help="Width of the video stream")
    parser.add_argument("--height", type=int, default=480, help="Height of the video stream")
    parser.add_argument("--log", type=str, default="INFO", help="Logging level")
    parser.add_argument("--output", type=str, help="Output file to save detected cubes")
    args = parser.parse_args()

    logging.basicConfig(level=args.log.upper())

    run_realsense_cube_detector(min_area=args.min_area, width=args.width, height=args.height, log_level=args.log, output_file=args.output)

if __name__ == "__main__":
    main()
