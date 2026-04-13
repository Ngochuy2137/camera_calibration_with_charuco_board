# camera_calibration_with_charuco_board

A tool for **camera calibration** using a [ChArUco board](https://docs.opencv.org/4.x/df/d4a/tutorial_charuco_detection.html) (a combination of a chessboard and ArUco markers). ChArUco boards offer sub-pixel corner accuracy and work well even with partial board visibility, making them well-suited for high-quality intrinsic camera calibration.

---

## Features

- Calibrate a camera's intrinsic parameters (camera matrix and distortion coefficients) using a ChArUco board.
- Flexible image input — choose the source that best fits your workflow:

| Image Source | Description |
|---|---|
| **Image files** | Load a folder of pre-captured calibration images from disk. |
| **USB camera** | Capture frames in real-time directly from a connected USB/webcam. |
| **ROS 2 topic** | Subscribe to a ROS 2 image topic and collect frames from a live camera feed. |

---

## Requirements

- Python 3
- OpenCV (with `opencv-contrib-python` for ArUco support)
- *(Optional, for ROS 2 topic source)* ROS 2 (Humble or newer) with `cv_bridge`

---

## Usage

### 1. Image files

Collect a set of calibration images containing the ChArUco board and point the tool at the directory:

```bash
python3 calibrate.py --source images --image_dir /path/to/images
```

### 2. USB camera

Capture calibration frames interactively from a USB or built-in webcam:

```bash
python3 calibrate.py --source usb --camera_id 0
```

Press a key to save a frame and `q` to finish capturing and run calibration.

### 3. ROS 2 topic

Subscribe to a ROS 2 image topic and collect frames from the live stream:

```bash
python3 calibrate.py --source ros2 --topic /camera/image_raw
```

---

## Output

After a successful calibration, the tool saves:

- **Camera matrix** – focal lengths and principal point.
- **Distortion coefficients** – radial and tangential distortion parameters.

Results are written to a YAML / JSON file that can be loaded directly by OpenCV or ROS 2 camera drivers.

---

## License

This project is licensed under the terms of the [LICENSE](LICENSE) file included in this repository.
