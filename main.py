# main.py

import argparse
import logging
from realsense_cube_detector import run_realsense_object_detector

def main():
    parser = argparse.ArgumentParser(description="Run RealSense Object Detector")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model file (.h5)")
    parser.add_argument("--labels", type=str, required=True, help="Path to the label file")
    parser.add_argument("--width", type=int, default=640, help="Width of the video stream")
    parser.add_argument("--height", type=int, default=480, help="Height of the video stream")
    parser.add_argument("--log", type=str, default="INFO", help="Logging level")
    parser.add_argument("--output", type=str, help="Output file to save detected objects")
    args = parser.parse_args()

    logging.basicConfig(level=args.log.upper())

    detected_objects = run_realsense_object_detector(
        model_path=args.model,
        label_path=args.labels,
        resolution=(args.width, args.height),
        log_level=args.log,
        output_file=args.output
    )
    print(f"Detected objects: {detected_objects}")

if __name__ == "__main__":
    main()
