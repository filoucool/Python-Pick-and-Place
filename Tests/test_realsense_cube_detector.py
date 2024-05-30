import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import cv2
from detector import RealSenseCubeDetector  # Adjust import path


class TestRealSenseCubeDetector(unittest.TestCase):

    @patch('pyrealsense2.pipeline')
    @patch('pyrealsense2.config')
    def test_initialization(self, mock_config, mock_pipeline):
        mock_pipeline_instance = MagicMock()
        mock_pipeline.return_value = mock_pipeline_instance

        detector = RealSenseCubeDetector(min_area=150, resolution=(1280, 720), output_file='test_output.json')

        self.assertEqual(detector.min_area, 150)
        self.assertEqual(detector.pipeline, mock_pipeline_instance)
        self.assertEqual(detector.output_file, 'test_output.json')
        self.assertEqual(detector.config, mock_config.return_value)

    def test_define_color_ranges(self):
        detector = RealSenseCubeDetector()
        color_ranges = detector.define_color_ranges()

        expected_ranges = {
            'red': (np.array([0, 120, 70]), np.array([10, 255, 255])),
            'green': (np.array([40, 40, 40]), np.array([80, 255, 255])),
            'blue': (np.array([100, 150, 150]), np.array([140, 255, 255])),
            'yellow': (np.array([20, 100, 100]), np.array([30, 255, 255])),
            'cyan': (np.array([80, 100, 100]), np.array([100, 255, 255])),
            'orange': (np.array([10, 100, 100]), np.array([25, 255, 255])),
            'pink': (np.array([145, 60, 65]), np.array([165, 255, 255]))
        }

        self.assertEqual(color_ranges.keys(), expected_ranges.keys())
        for key in expected_ranges.keys():
            np.testing.assert_array_equal(color_ranges[key][0], expected_ranges[key][0])
            np.testing.assert_array_equal(color_ranges[key][1], expected_ranges[key][1])

    @patch('pyrealsense2.pipeline')
    @patch('cv2.imshow')
    @patch('cv2.waitKey', return_value=ord('q'))
    def test_detect_cubes(self, mock_waitKey, mock_imshow, mock_pipeline):
        detector = RealSenseCubeDetector()
        detector.pipeline = MagicMock()
        detector.pipeline.wait_for_frames = MagicMock()

        mock_frames = MagicMock()
        mock_depth_frame = MagicMock()
        mock_color_frame = MagicMock()

        mock_frames.get_depth_frame.return_value = mock_depth_frame
        mock_frames.get_color_frame.return_value = mock_color_frame
        mock_color_frame.get_data = MagicMock(return_value=np.zeros((480, 640, 3), np.uint8))

        detector.pipeline.wait_for_frames.return_value = mock_frames

        detected_cubes = detector.detect_cubes()

        self.assertEqual(detected_cubes, [])
        called_image = mock_imshow.call_args[0][1]
        self.assertTrue(np.array_equal(called_image, np.zeros((480, 640, 3), np.uint8)))


if __name__ == "__main__":
    unittest.main()
