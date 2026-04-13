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
`opencv-python` (or system OpenCV) with `cv2.aruco`. ROS image mode also needs `rclpy`, `sensor_msgs`, and `cv_bridge` (source your ROS workspace before running).

**Recommendation:** Use the **same resolution and camera settings** you use on the robot so intrinsics match deployment.

## Run calibration
Input sources (`--source`):

| Mode | Description |
|------|-------------|
| `file` | Still images from disk (glob patterns). Default `--source`. |
| `usb` | Live capture via `cv2.VideoCapture` (`--camera_id`, default `0`). |
| `ros2` | Subscribe to `sensor_msgs/Image` (`--topic`, default `/camera/image_raw`). |

Examples (replace measured sizes with yours):

```bash
# From saved images
ros2 run camera calibrate_camera_node.py --source file 'calib_frames/*.jpg' \
    --square_length 3.88 --marker_size 2.8 --preview topic

# USB camera (on Linux/Jetson, default --usb_backend auto uses V4L2 first to avoid GStreamer crashes)
ros2 run camera calibrate_camera_node.py --source usb --usb_backend v4l2 \
    --square_length 3.88 --marker_size 2.8 --preview topic

# ROS 2 topic (adjust --topic to your driver)
ros2 run camera calibrate_camera_node.py --source ros2 --topic /camera/image_raw \
    --square_length 3.88 --marker_size 2.8 --preview topic
```

## Preview and controls (`usb` / `ros2`):**

| `--preview` | Behavior |
|-------------|----------|
| `auto` (default) | OpenCV window if `DISPLAY` is set; otherwise headless (see below). |
| `gui` | Always use OpenCV window (`c` / `q` / ESC). |
| `topic` | Always headless: publish overlay JPEGs to `sensor_msgs/CompressedImage` on `--preview_topic` (default `/charuco_calib/preview/compressed`) and use **Trigger services** on node `--ros_node_name` (default `charuco_calib`): `capture_frame`, `finish_calibration`, `abort_calibration`. |

GUI: **`c`** or **Enter** (with the OpenCV window focused) captures the current frame; **`q`** finishes and calibrates; **ESC** aborts.

If the process has a **terminal (TTY)**—for example an interactive `ros2 run ...` over SSH—you can type **Enter** alone on a new line to capture (after holding the board steady), **`q`** + Enter to finish, or **`abort`** + Enter to quit. You should see `[calib] stdin -> …` in the terminal when a line is read. **`ros2 launch` often does not attach a TTY** to the process, so Enter does nothing there; use **Trigger services** instead. A background ROS spinner handles service calls even while `VideoCapture.read()` is blocking.

Headless service examples:

```bash
ros2 service call /charuco_calib/capture_frame std_srvs/srv/Trigger {}
ros2 service call /charuco_calib/finish_calibration std_srvs/srv/Trigger {}
ros2 service call /charuco_calib/abort_calibration std_srvs/srv/Trigger {}
```

View preview: `ros2 run rqt_image_view rqt_image_view` → topic `/charuco_calib/preview/compressed` (or your `--preview_topic`).

**How to move the board (what “15–30 varied views” means):**  
Goal: each saved frame should show the board **clearly**, with **many ChArUco corners** visible, but from **different combinations** of distance, angle, and placement in the image. That gives the solver enough constraints for intrinsics and distortion—not random blur.

- **Distance:** Capture some frames **closer** (board large in the image) and some **farther** (board smaller), still sharp and well lit. Avoid extreme zoom where markers are tiny.
- **Tilt / orientation:** Tilt the board so its plane is **not always parallel** to the image plane: slight **left/right roll**, **top/bottom pitch**, and **yaw** (rotate the board like steering). You are *not* aiming for one perfect frontal pose every time.
- **Position in the frame:** Put the board toward the **left, center, right**, and **upper / lower** parts of the image so corners land in **different pixels** (important for lens distortion).
- **When capturing:** **Hold steady** for the moment you press `c` or call `capture_frame`—motion blur ruins detection. Good lighting, no glare on the print, full board in view when possible.

Roughly **15–30** accepted frames is typical; more can help if RMS is high. The script **skips** frames with too few corners (`--min_corners`, default `6`).

Use `calibrate_camera_node.py --help` for all options (`--debug_images` / `--debug_dir`, cooldown, board size, `--preview_jpeg_quality`, etc.).
## License

This project is licensed under the terms of the [LICENSE](LICENSE) file included in this repository.
