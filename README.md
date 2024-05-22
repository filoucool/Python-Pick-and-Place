# Python Bin Picking

Python Bin Picking is an application designed to automate the identification and picking of objects in a bin using an Intel RealSense camera. Leveraging computer vision techniques, the script detects specific objects by their color and size, and guides robotic mechanisms to pick these objects efficiently.

## Features

- Real-time cube detection using an Intel RealSense camera.
 <img src="https://github.com/filoucool/Python-Pick-and-Place/assets/25182703/1f05f9b2-e56b-4c64-bc03-1874d444d731" height="300"/>
- Detection of multiple colors including red, green, blue, yellow, cyan, orange, and pink.
- Adjustable minimum area for detected cubes to filter out noise.
- Output detected cubes' information to a JSON file.

## Requirements

- Python 3.6 or higher
- OpenCV (`cv2`)
- `pyrealsense2`
- NumPy

## Installation

Ensure you have the required libraries installed. You can install them using pip:

```bash
pip install numpy opencv-python pyrealsense2
```

## Installation
```
python RealSenseBinPicking.py --object_size 150 --width 640 --height 480 --log INFO --output_mode json
```

## Parameters
- object_size: Minimum contour area to consider a shape suitable for picking (default is 150 pixels squared).
- width and --height: Resolution of the video stream (default is 640x480).
- log: Logging level (default is INFO).
- output_mode: Specifies the output format (e.g., json for integration with robotics systems).

## Configuration
- Color Ranges: Define the color ranges in HSV for object detection based on the specific objects in your bin.
- Process Frame: Adjust image processing steps to optimize detection under varying lighting and bin conditions.

## My dev Setup
- Robot Arm: Ufactory xArm 850
<img src="https://github.com/filoucool/Python-Pick-and-Place/assets/25182703/b76ca0e1-fe87-4766-9bd7-a273b2cdcd4e" height="300"/>

- Gripper:
  Custom Made
- Camera:
    Intel RealSense D435i

## Output
The detected objects will be displayed in a window with their respective bounding boxes and identified color labels. If output_mode is set to json, the script will output the coordinates and dimensions of detected objects in a JSON format for use by robotic picking systems.

## Contributing
Contributions to enhance the functionality or adapt the script to specific picking tasks or different environments are welcome. Please follow the existing code style and add comprehensive comments.

## License
This project is open source and available under the MIT License.
