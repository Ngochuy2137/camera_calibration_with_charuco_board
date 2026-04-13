#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import json
import os
import sys
import threading
import time
from collections import OrderedDict
from datetime import datetime
from typing import Optional, Tuple

import cv2
import numpy as np


ARUCO_DICTS = {
    'DICT_4X4_50': cv2.aruco.DICT_4X4_50,
    'DICT_4X4_100': cv2.aruco.DICT_4X4_100,
    'DICT_4X4_250': cv2.aruco.DICT_4X4_250,
    'DICT_4X4_1000': cv2.aruco.DICT_4X4_1000,
    'DICT_5X5_50': cv2.aruco.DICT_5X5_50,
    'DICT_5X5_100': cv2.aruco.DICT_5X5_100,
    'DICT_5X5_250': cv2.aruco.DICT_5X5_250,
    'DICT_5X5_1000': cv2.aruco.DICT_5X5_1000,
    'DICT_6X6_50': cv2.aruco.DICT_6X6_50,
    'DICT_6X6_100': cv2.aruco.DICT_6X6_100,
    'DICT_6X6_250': cv2.aruco.DICT_6X6_250,
    'DICT_6X6_1000': cv2.aruco.DICT_6X6_1000,
    'DICT_7X7_50': cv2.aruco.DICT_7X7_50,
    'DICT_7X7_100': cv2.aruco.DICT_7X7_100,
    'DICT_7X7_250': cv2.aruco.DICT_7X7_250,
    'DICT_7X7_1000': cv2.aruco.DICT_7X7_1000,
    'DICT_ARUCO_ORIGINAL': cv2.aruco.DICT_ARUCO_ORIGINAL,
}

for name in ['DICT_APRILTAG_16h5', 'DICT_APRILTAG_25h9', 'DICT_APRILTAG_36h10', 'DICT_APRILTAG_36h11']:
    if hasattr(cv2.aruco, name):
        ARUCO_DICTS[name] = getattr(cv2.aruco, name)


def get_aruco_dict(dict_name: str):
    if dict_name not in ARUCO_DICTS:
        raise ValueError(f'Unknown dictionary: {dict_name}')
    dict_id = ARUCO_DICTS[dict_name]

    if hasattr(cv2.aruco, 'getPredefinedDictionary'):
        return cv2.aruco.getPredefinedDictionary(dict_id)
    return cv2.aruco.Dictionary_get(dict_id)


def create_charuco_board(squares_x: int, squares_y: int, square_len: float, marker_len: float, aruco_dict):
    """
    Prefer CharucoBoard_create when available. On Ubuntu/Jetson OpenCV 4.6 Python builds,
    cv2.aruco.CharucoBoard (new wrapper) + interpolateCornersCharuco can segfault once markers
    are detected; the legacy board object does not.
    Returns (board, allow_charuco_detector).
    """
    if hasattr(cv2.aruco, 'CharucoBoard_create'):
        board = cv2.aruco.CharucoBoard_create(
            squares_x,
            squares_y,
            square_len,
            marker_len,
            aruco_dict
        )
        return board, False

    if hasattr(cv2.aruco, 'CharucoBoard'):
        board = cv2.aruco.CharucoBoard(
            (squares_x, squares_y),
            square_len,
            marker_len,
            aruco_dict
        )
        return board, True

    raise RuntimeError('Your cv2.aruco does not support CharucoBoard.')


def create_detector(board, allow_charuco_detector: bool):
    if not allow_charuco_detector:
        return None
    if hasattr(cv2.aruco, 'CharucoDetector'):
        return cv2.aruco.CharucoDetector(board)
    return None


def detect_charuco_compatible(gray, board, aruco_dict, detector=None):
    """
    Return:
        charuco_corners, charuco_ids, marker_corners, marker_ids
    """
    if detector is not None:
        corners, ids, marker_corners, marker_ids = detector.detectBoard(gray)
        return corners, ids, marker_corners, marker_ids

    if hasattr(cv2.aruco, 'DetectorParameters_create'):
        params = cv2.aruco.DetectorParameters_create()
    else:
        params = cv2.aruco.DetectorParameters()

    marker_corners, marker_ids, _rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)

    if marker_ids is None or len(marker_ids) == 0:
        return None, None, marker_corners, marker_ids

    retval = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, gray, board)

    if len(retval) == 3:
        maybe_count, charuco_corners, charuco_ids = retval
        if isinstance(maybe_count, (int, float)) and maybe_count <= 0:
            return None, None, marker_corners, marker_ids
    elif len(retval) == 2:
        charuco_corners, charuco_ids = retval
    else:
        raise RuntimeError('Unexpected return from interpolateCornersCharuco')

    return charuco_corners, charuco_ids, marker_corners, marker_ids


def match_image_points_compatible(board, charuco_corners, charuco_ids):
    if hasattr(board, 'matchImagePoints'):
        return board.matchImagePoints(charuco_corners, charuco_ids)
    return None, None


def flatten_ids(ids) -> np.ndarray:
    ids = np.asarray(ids)
    return ids.reshape(-1, 1).astype(np.int32)


def ensure_bgr(img):
    if img is None:
        return None
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def ensure_gray(img):
    if img is None:
        return None
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def save_debug_image(debug_dir: Optional[str], name: str, gray, marker_corners, marker_ids, charuco_corners, charuco_ids):
    if not debug_dir:
        return None
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if marker_ids is not None and len(marker_ids) > 0:
        cv2.aruco.drawDetectedMarkers(vis, marker_corners, marker_ids)

    if charuco_ids is not None and len(charuco_ids) > 0:
        cv2.aruco.drawDetectedCornersCharuco(vis, charuco_corners, charuco_ids)

    out_path = os.path.join(debug_dir, f'{name}_detected.png')
    cv2.imwrite(out_path, vis)
    return out_path


def draw_live_overlay(frame_bgr, marker_corners, marker_ids, charuco_corners, charuco_ids, info_text):
    vis = frame_bgr.copy()

    if marker_ids is not None and len(marker_ids) > 0:
        cv2.aruco.drawDetectedMarkers(vis, marker_corners, marker_ids)

    if charuco_ids is not None and len(charuco_ids) > 0:
        cv2.aruco.drawDetectedCornersCharuco(vis, charuco_corners, charuco_ids)

    y = 30
    for line in info_text:
        cv2.putText(vis, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        y += 28

    return vis


class CalibrationAccumulator:
    def __init__(self, board, min_corners: int, debug_dir: Optional[str], save_captures_dir: Optional[str]):
        self.board = board
        self.min_corners = min_corners
        self.debug_dir = debug_dir
        self.save_captures_dir = save_captures_dir

        self.all_obj_points = []
        self.all_img_points = []
        self.all_charuco_corners = []
        self.all_charuco_ids = []
        self.image_size: Optional[Tuple[int, int]] = None
        self.kept = 0

    def try_add_frame(self, frame_bgr, gray, frame_name, marker_corners, marker_ids, charuco_corners, charuco_ids):
        if gray is None:
            print(f'[SKIP] {frame_name}: gray image is None')
            return False

        current_size = (gray.shape[1], gray.shape[0])
        if self.image_size is None:
            self.image_size = current_size
        elif self.image_size != current_size:
            print(f'[SKIP] {frame_name}: different image size {current_size}, expected {self.image_size}')
            return False

        if charuco_ids is None or charuco_corners is None:
            print(f'[SKIP] {frame_name}: no ChArUco corners')
            save_debug_image(self.debug_dir, frame_name, gray, marker_corners, marker_ids, [], [])
            return False

        num_corners = len(charuco_ids)
        if num_corners < self.min_corners:
            print(f'[SKIP] {frame_name}: too few corners ({num_corners})')
            save_debug_image(self.debug_dir, frame_name, gray, marker_corners, marker_ids, charuco_corners, charuco_ids)
            return False

        frame_obj_points, frame_img_points = match_image_points_compatible(self.board, charuco_corners, charuco_ids)

        if frame_obj_points is not None and frame_img_points is not None:
            if len(frame_obj_points) < 4 or len(frame_img_points) < 4:
                print(f'[SKIP] {frame_name}: not enough matched points')
                save_debug_image(self.debug_dir, frame_name, gray, marker_corners, marker_ids, charuco_corners, charuco_ids)
                return False

            self.all_obj_points.append(np.asarray(frame_obj_points, dtype=np.float32))
            self.all_img_points.append(np.asarray(frame_img_points, dtype=np.float32))

        self.all_charuco_corners.append(np.asarray(charuco_corners, dtype=np.float32))
        self.all_charuco_ids.append(flatten_ids(charuco_ids))

        self.kept += 1
        save_debug_image(self.debug_dir, frame_name, gray, marker_corners, marker_ids, charuco_corners, charuco_ids)

        if self.save_captures_dir is not None:
            os.makedirs(self.save_captures_dir, exist_ok=True)
            raw_path = os.path.join(self.save_captures_dir, f'{frame_name}.png')
            cv2.imwrite(raw_path, frame_bgr)

        print(f'[OK] {frame_name}: kept, corners={num_corners}, total={self.kept}')
        return True


def calibrate_from_accumulator(acc: CalibrationAccumulator, board):
    if acc.kept < 5:
        raise RuntimeError('Too few valid images/frames. Try collecting 15-30 from varied viewpoints.')

    if acc.image_size is None:
        raise RuntimeError('No valid image size available.')

    if len(acc.all_obj_points) >= 5 and len(acc.all_img_points) >= 5:
        rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            acc.all_obj_points,
            acc.all_img_points,
            acc.image_size,
            None,
            None
        )
    else:
        if not hasattr(cv2.aruco, 'calibrateCameraCharuco'):
            raise RuntimeError('Neither matchImagePoints nor calibrateCameraCharuco is available.')

        rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
            acc.all_charuco_corners,
            acc.all_charuco_ids,
            board,
            acc.image_size,
            None,
            None
        )

    return rms, camera_matrix, dist_coeffs, rvecs, tvecs


def save_calibration_yaml(path, image_size, args, rms, camera_matrix, dist_coeffs):
    path = os.path.expanduser(path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    fs.write('image_width', image_size[0])
    fs.write('image_height', image_size[1])
    fs.write('board_squares_x', args.width)
    fs.write('board_squares_y', args.height)
    fs.write('square_length', args.square_length)
    fs.write('marker_size', args.marker_size)
    fs.write('aruco_dict', args.aruco_dict)
    fs.write('source', args.source)
    fs.write('rms', float(rms))
    fs.write('camera_matrix', camera_matrix)
    fs.write('dist_coeffs', dist_coeffs)
    fs.release()


def save_calibration_json(
    path: str,
    image_size: Tuple[int, int],
    camera_matrix,
    dist_coeffs,
    rms: float,
    requested_width: Optional[int] = None,
    requested_height: Optional[int] = None,
    requested_fps: int = 30,
):
    """
    Same layout as legacy CameraCalibrator.save_params_to_json (camera_config.json).
    image_size is (width, height).
    """
    img_w, img_h = int(image_size[0]), int(image_size[1])
    rw = int(requested_width) if requested_width is not None else img_w
    rh = int(requested_height) if requested_height is not None else img_h
    dt_string = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    data = OrderedDict([
        ('calibration_time', dt_string),
        ('image_width', img_w),
        ('image_height', img_h),
        ('requested_width', rw),
        ('requested_height', rh),
        ('requested_fps', int(requested_fps)),
        ('camera_matrix', np.asarray(camera_matrix).tolist()),
        ('distortion_coefficients', np.asarray(dist_coeffs).tolist()),
        ('avg_reprojection_error', float(rms)),
    ])
    parent = os.path.dirname(os.path.expanduser(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    out_path = os.path.expanduser(path)
    with open(out_path, 'w') as f:
        json.dump(data, f, indent=4)
    return out_path


def save_ros_usb_cam_camera_info_yaml(
    path: str,
    image_size: Tuple[int, int],
    camera_matrix,
    dist_coeffs,
    camera_name: str = 'usb_cam_color',
    distortion_model: str = 'plumb_bob',
    rms: Optional[float] = None,
) -> str:
    """
    Write ROS camera_calibration-style YAML for usb_cam `camera_info_url`
    (sensor_msgs/CameraInfo). Mono pinhole: rectification_matrix = I, projection_matrix from K
    with the fourth column zero (no stereo baseline).
    """
    path = os.path.expanduser(path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    w, h = int(image_size[0]), int(image_size[1])
    K = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)

    d = np.asarray(dist_coeffs, dtype=np.float64).ravel()
    if distortion_model == 'plumb_bob':
        if d.size < 5:
            d = np.pad(d, (0, 5 - d.size))
        elif d.size > 5:
            d = d[:5].copy()

    def _fmt_row(vals):
        return '[' + ', '.join(f'{float(x):.6f}' for x in vals) + ']'

    r_data = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    # Mono: P[:, :3] = K (includes skew if any); last column zero (no stereo Tx).
    p_data = [
        float(K[0, 0]),
        float(K[0, 1]),
        float(K[0, 2]),
        0.0,
        float(K[1, 0]),
        float(K[1, 1]),
        float(K[1, 2]),
        0.0,
        float(K[2, 0]),
        float(K[2, 1]),
        float(K[2, 2]),
        0.0,
    ]
    k_data = [float(K[i, j]) for i in range(3) for j in range(3)]

    lines = [
        '# ROS camera_calibration YAML for usb_cam camera_info_url (sensor_msgs/CameraInfo).',
        '# Generated by calibrate_camera_node.py. Mono: rectification_matrix = I; P from K.',
    ]
    if rms is not None:
        lines.append(f'# avg_reprojection_error: {float(rms):.6f}')
    lines.extend(
        [
            f'image_width: {w}',
            f'image_height: {h}',
            f'camera_name: {camera_name}',
            'camera_matrix:',
            '  rows: 3',
            '  cols: 3',
            f'  data: {_fmt_row(k_data)}',
            f'distortion_model: {distortion_model}',
            'distortion_coefficients:',
            '  rows: 1',
            f'  cols: {int(d.size)}',
            f'  data: {_fmt_row(d)}',
            'rectification_matrix:',
            '  rows: 3',
            '  cols: 3',
            f'  data: {_fmt_row(r_data)}',
            'projection_matrix:',
            '  rows: 3',
            '  cols: 4',
            f'  data: {_fmt_row(p_data)}',
            '',
        ]
    )
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    return path


def run_file_source(args, board, aruco_dict, detector, acc: CalibrationAccumulator):
    image_paths = []
    for pattern in args.images:
        image_paths.extend(sorted(glob.glob(pattern)))
    image_paths = sorted(set(image_paths))

    if not image_paths:
        raise RuntimeError('No images found for file source.')

    for idx, path in enumerate(image_paths):
        frame_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            print(f'[SKIP] Cannot read image: {path}')
            continue

        gray = ensure_gray(frame_bgr)

        try:
            charuco_corners, charuco_ids, marker_corners, marker_ids = detect_charuco_compatible(
                gray, board, aruco_dict, detector
            )
        except Exception as e:
            print(f'[SKIP] Detection error on {path}: {e}')
            continue

        frame_name = os.path.splitext(os.path.basename(path))[0]
        acc.try_add_frame(frame_bgr, gray, frame_name, marker_corners, marker_ids, charuco_corners, charuco_ids)


def use_gui_preview(args) -> bool:
    """GUI (OpenCV window) vs headless (ROS CompressedImage + Trigger services)."""
    if args.preview == 'gui':
        return True
    if args.preview == 'topic':
        return False
    return bool(os.environ.get('DISPLAY', '').strip())


class HeadlessCmdFlags:
    __slots__ = ('capture', 'finish', 'abort')

    def __init__(self):
        self.capture = False
        self.finish = False
        self.abort = False


def _register_charuco_calib_services(node, flags: HeadlessCmdFlags):
    from std_srvs.srv import Trigger

    def _capture_cb(_req, resp):
        flags.capture = True
        node.get_logger().info('Service capture_frame received (queued for main loop)')
        print('[calib] service capture_frame -> queued', flush=True)
        resp.success = True
        resp.message = 'Capture queued'
        return resp

    def _finish_cb(_req, resp):
        flags.finish = True
        node.get_logger().info('Service finish_calibration received (queued)')
        print('[calib] service finish_calibration -> queued', flush=True)
        resp.success = True
        resp.message = 'Finish queued'
        return resp

    def _abort_cb(_req, resp):
        flags.abort = True
        node.get_logger().info('Service abort_calibration received (queued)')
        print('[calib] service abort_calibration -> queued', flush=True)
        resp.success = True
        resp.message = 'Abort queued'
        return resp

    # Use ~/ so services resolve under this node (e.g. /charuco_calib/capture_frame).
    # A bare 'capture_frame' expands to /capture_frame when the node namespace is '/'.
    node.create_service(Trigger, '~/capture_frame', _capture_cb)
    node.create_service(Trigger, '~/finish_calibration', _finish_cb)
    node.create_service(Trigger, '~/abort_calibration', _abort_cb)


def _publish_preview_compressed(node, publisher, vis_bgr, jpeg_quality: int):
    from sensor_msgs.msg import CompressedImage

    msg = CompressedImage()
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.format = 'jpeg'
    ok, buf = cv2.imencode('.jpg', vis_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    if not ok:
        return
    msg.data = buf.tobytes()
    publisher.publish(msg)


def _drain_rclpy_node(node, max_iter: int = 24):
    """Process pending subscriptions and service calls so clients do not hang."""
    import rclpy
    from rclpy.executors import ExternalShutdownException

    for _ in range(max_iter):
        try:
            rclpy.spin_once(node, timeout_sec=0.001)
        except ExternalShutdownException:
            break


def _start_rclpy_background_spin(node, stop_event: threading.Event, name: str = 'charuco_calib_spin'):
    """
    While the main thread blocks on cap.read() or OpenCV, keep servicing DDS so
    Trigger service clients return immediately.
    """

    def _loop():
        import rclpy

        while not stop_event.is_set():
            try:
                if not rclpy.ok():
                    break
            except Exception:
                break
            _drain_rclpy_node(node, 12)
            time.sleep(0.004)

    th = threading.Thread(target=_loop, daemon=True, name=name)
    th.start()
    return th


def _consume_headless_service_flags(flags: HeadlessCmdFlags) -> int:
    """
    Map queued Trigger service requests to virtual keys.
    Must NOT call rclpy.spin_once here: a background thread already spins the node.
    """
    if flags.abort:
        flags.abort = False
        return 27
    if flags.finish:
        flags.finish = False
        return ord('q')
    if flags.capture:
        flags.capture = False
        return ord('c')
    return 0


def _tty_line_command_if_any() -> int:
    """
    Non-blocking: if stdin is a TTY and a full line is available (user pressed Enter),
    consume it and map to a virtual key. Returns 0 if nothing to read.
    Empty line -> capture (same as 'c'). Useful when the OpenCV window has no focus.
    """
    if not sys.stdin.isatty():
        return 0
    if sys.platform == 'win32':
        return 0
    try:
        import select
    except ImportError:
        return 0
    try:
        readable, _, _ = select.select([sys.stdin], [], [], 0)
        if not readable:
            return 0
        line = sys.stdin.readline()
    except (ValueError, OSError):
        return 0
    s = line.strip().lower()
    if s == '':
        return ord('c')
    if s in ('q', 'quit', 'finish'):
        return ord('q')
    if s in ('abort', 'a'):
        return 27
    if s in ('c', 'capture', 'cap'):
        return ord('c')
    return 0


def _merge_live_keys(key: int) -> int:
    """Map GUI Enter to capture; merge TTY line commands (Enter / q / abort)."""
    if key in (13, 10):
        return ord('c')
    tty_key = _tty_line_command_if_any()
    if tty_key != 0:
        label = {ord('c'): 'capture', ord('q'): 'finish', 27: 'abort'}.get(tty_key, str(tty_key))
        print(f'[calib] stdin -> {label}', flush=True)
        return tty_key
    return key


class CharucoHeadlessUsbNode:
    """rclpy node: preview publisher + Trigger services (USB, no display)."""

    def __init__(self, node_name: str, preview_topic: str, jpeg_quality: int):
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import CompressedImage

        rclpy.init(args=None)
        self._node = Node(node_name)
        self._flags = HeadlessCmdFlags()
        self._pub = self._node.create_publisher(CompressedImage, preview_topic, 1)
        self._jpeg_quality = jpeg_quality
        _register_charuco_calib_services(self._node, self._flags)
        self._node.get_logger().info(
            f'Headless USB: preview topic {preview_topic} | '
            f'services /{node_name}/capture_frame, /{node_name}/finish_calibration, /{node_name}/abort_calibration'
        )
        self._shutdown_done = False
        self._spin_stop = threading.Event()
        self._spin_thread = _start_rclpy_background_spin(self._node, self._spin_stop, 'charuco_usb_spin')

    @property
    def node(self):
        return self._node

    @property
    def flags(self):
        return self._flags

    def publish_preview(self, vis_bgr):
        _publish_preview_compressed(self._node, self._pub, vis_bgr, self._jpeg_quality)

    def poll_key(self):
        return _consume_headless_service_flags(self._flags)

    def shutdown(self):
        import rclpy

        if self._shutdown_done:
            return
        self._shutdown_done = True
        self._spin_stop.set()
        try:
            self._spin_thread.join(timeout=2.0)
        except Exception:
            pass
        try:
            self._node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


class CharucoCalibRos2ImageNode:
    """Subscribe to sensor_msgs/Image; optional headless preview + services."""

    def __init__(self, image_topic: str, use_gui: bool, preview_topic: str, jpeg_quality: int, node_name: str):
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import CompressedImage, Image
        from cv_bridge import CvBridge

        rclpy.init(args=None)
        self._node = Node(node_name)
        self.use_gui = use_gui
        self._bridge = CvBridge()
        self.latest_frame = None
        self._jpeg_quality = jpeg_quality
        self._pub = None
        self._flags = HeadlessCmdFlags()
        self._node.create_subscription(Image, image_topic, self._image_cb, 10)
        if not use_gui:
            self._pub = self._node.create_publisher(CompressedImage, preview_topic, 1)
            _register_charuco_calib_services(self._node, self._flags)
            self._node.get_logger().info(
                f'Headless ROS2: input {image_topic} preview {preview_topic} | '
                f'services ~capture_frame ~finish_calibration ~abort_calibration (node {node_name})'
            )
        else:
            self._node.get_logger().info(f'ROS2 GUI: subscribing {image_topic}')
        self._shutdown_done = False
        self._spin_stop = None
        self._spin_thread = None
        if not use_gui:
            self._spin_stop = threading.Event()
            self._spin_thread = _start_rclpy_background_spin(
                self._node, self._spin_stop, 'charuco_ros2_spin'
            )

    @property
    def flags(self):
        return self._flags

    def _image_cb(self, msg):
        try:
            if msg.encoding in ('mono8', '8UC1'):
                gray = self._bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
                self.latest_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            else:
                self.latest_frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as err:
            self._node.get_logger().warn(f'cv_bridge conversion failed: {err}')

    @property
    def node(self):
        return self._node

    def publish_preview(self, vis_bgr):
        if self._pub is None:
            return
        _publish_preview_compressed(self._node, self._pub, vis_bgr, self._jpeg_quality)

    def poll_key(self):
        if self.use_gui:
            return cv2.waitKey(1) & 0xFF
        return _consume_headless_service_flags(self._flags)

    def spin_once(self):
        import rclpy

        rclpy.spin_once(self._node, timeout_sec=0.02)

    def shutdown(self):
        import rclpy

        if getattr(self, '_shutdown_done', False):
            return
        self._shutdown_done = True
        if self._spin_stop is not None:
            self._spin_stop.set()
        if self._spin_thread is not None:
            try:
                self._spin_thread.join(timeout=2.0)
            except Exception:
                pass
        try:
            self._node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


def open_usb_camera(camera_id: int, backend: str):
    """
    Open a USB camera. On many Linux ARM boards (e.g. Jetson), the default OpenCV
    backend picks GStreamer and can warn or segfault; V4L2 is usually stable.
    backend: 'auto' | 'v4l2' | 'default'
    """
    v4l2 = getattr(cv2, 'CAP_V4L2', None)

    if backend == 'v4l2':
        if v4l2 is None:
            raise RuntimeError('CAP_V4L2 is not available in this OpenCV build')
        cap = cv2.VideoCapture(camera_id, v4l2)
        return cap

    if backend == 'default':
        return cv2.VideoCapture(camera_id)

    # auto
    if sys.platform.startswith('linux') and v4l2 is not None:
        cap = cv2.VideoCapture(camera_id, v4l2)
        if cap.isOpened():
            return cap
        cap.release()
        print('[WARN] USB camera: V4L2 failed to open; retrying default backend', flush=True)
    return cv2.VideoCapture(camera_id)


def run_usb_source(args, board, aruco_dict, detector, acc: CalibrationAccumulator, use_gui: bool):
    cap = open_usb_camera(args.camera_id, args.usb_backend)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open USB camera index {args.camera_id}')

    headless = None
    if not use_gui:
        try:
            headless = CharucoHeadlessUsbNode(
                args.ros_node_name,
                args.preview_topic,
                args.preview_jpeg_quality,
            )
        except Exception as e:
            cap.release()
            raise RuntimeError(
                'Headless USB mode requires rclpy, sensor_msgs, std_srvs. '
                'Source your ROS 2 workspace, or use --preview gui with DISPLAY set.'
            ) from e

    print('USB mode:')
    if use_gui:
        print('  c or Enter (in window): capture current frame if detection is good')
        print('  q: finish and calibrate')
        print('  ESC: quit without calibrating')
    else:
        print(f'  preview: {args.preview_topic} (sensor_msgs/CompressedImage, jpeg)')
        print(f'  ros2 service call /{args.ros_node_name}/capture_frame std_srvs/srv/Trigger "{{}}"')
        print(f'  ros2 service call /{args.ros_node_name}/finish_calibration std_srvs/srv/Trigger "{{}}"')
        print(f'  ros2 service call /{args.ros_node_name}/abort_calibration std_srvs/srv/Trigger "{{}}"')
    if sys.stdin.isatty():
        print('  TTY (this terminal): Enter = capture | q + Enter = finish | abort + Enter = quit')
    elif not use_gui:
        print(
            '  Note: stdin is not a TTY (typical under `ros2 launch`). Enter here does nothing — '
            'use the Trigger services above, or run `ros2 run camera calibrate_camera_node.py ...` in this shell.',
            flush=True,
        )

    frame_idx = 0
    last_capture_time = 0.0

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                print('[WARN] Failed to read frame from USB camera')
                if headless is not None:
                    k = _merge_live_keys(headless.poll_key())
                else:
                    k = _merge_live_keys(0)
                if k == 27:
                    raise KeyboardInterrupt
                if k == ord('q'):
                    break
                continue

            gray = ensure_gray(frame_bgr)

            try:
                charuco_corners, charuco_ids, marker_corners, marker_ids = detect_charuco_compatible(
                    gray, board, aruco_dict, detector
                )
            except Exception as e:
                print(f'[WARN] Detection error: {e}')
                charuco_corners, charuco_ids, marker_corners, marker_ids = None, None, [], []

            num_corners = 0 if charuco_ids is None else len(charuco_ids)
            info_text = [
                f'source=usb camera_id={args.camera_id}',
                f'corners={num_corners} kept={acc.kept}',
                'c/Enter=capture  q=finish  esc=quit' if use_gui else 'TTY:Enter=capture | svc capture_frame',
            ]
            vis = draw_live_overlay(frame_bgr, marker_corners, marker_ids, charuco_corners, charuco_ids, info_text)

            if use_gui:
                cv2.imshow('charuco_live', vis)
                key = _merge_live_keys(cv2.waitKey(1) & 0xFF)
            else:
                headless.publish_preview(vis)
                key = _merge_live_keys(headless.poll_key())

            if key == 27:
                print('[calib] abort (ESC, abort service, or stdin)', flush=True)
                raise KeyboardInterrupt
            elif key == ord('q'):
                print('[calib] finish_calibration: running solve...', flush=True)
                break
            elif key == ord('c'):
                now = time.time()
                if now - last_capture_time < args.capture_cooldown:
                    print('[calib] capture ignored (cooldown)', flush=True)
                    continue
                frame_name = f'usb_{frame_idx:04d}'
                ok = acc.try_add_frame(frame_bgr, gray, frame_name, marker_corners, marker_ids, charuco_corners, charuco_ids)
                if ok:
                    print(f'[calib] captured {frame_name} (kept={acc.kept})', flush=True)
                    frame_idx += 1
                    last_capture_time = now
                else:
                    _rej = '[calib] capture rejected (corners/size)'
                    if args.debug_dir:
                        _rej += '; see debug_dir'
                    print(_rej, flush=True)
    finally:
        cap.release()
        if headless is not None:
            headless.shutdown()
        if use_gui:
            cv2.destroyAllWindows()


def run_ros2_source(args, board, aruco_dict, detector, acc: CalibrationAccumulator, use_gui: bool):
    try:
        import rclpy
    except Exception as e:
        raise RuntimeError(
            'ROS2 mode requires rclpy, sensor_msgs, cv_bridge, std_srvs. '
            'Make sure ROS2 environment is sourced.'
        ) from e

    try:
        host = CharucoCalibRos2ImageNode(
            args.topic,
            use_gui,
            args.preview_topic,
            args.preview_jpeg_quality,
            args.ros_node_name,
        )
    except Exception as e:
        raise RuntimeError(
            'ROS2 mode failed to start (rclpy, sensor_msgs, cv_bridge, std_srvs). '
            'Source your ROS 2 workspace.'
        ) from e

    print('ROS2 mode:')
    print(f'  input topic: {args.topic}')
    if use_gui:
        print('  c or Enter (in window): capture  q: finish  ESC: quit')
    else:
        print(f'  preview: {args.preview_topic}')
        print(f'  services: /{args.ros_node_name}/capture_frame, finish_calibration, abort_calibration')
    if sys.stdin.isatty():
        print('  TTY: Enter = capture | q + Enter = finish | abort + Enter = quit')
    elif not use_gui:
        print(
            '  Note: stdin is not a TTY (typical under `ros2 launch`). Enter here does nothing — '
            'use the Trigger services above, or run `ros2 run camera calibrate_camera_node.py ...` in this shell.',
            flush=True,
        )

    frame_idx = 0
    last_capture_time = 0.0

    try:
        while rclpy.ok():
            if use_gui:
                host.spin_once()

            if host.latest_frame is None:
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank, f'Waiting for topic: {args.topic}', (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                if use_gui:
                    cv2.imshow('charuco_live', blank)
                    key = _merge_live_keys(cv2.waitKey(1) & 0xFF)
                else:
                    host.publish_preview(blank)
                    key = _merge_live_keys(_consume_headless_service_flags(host.flags))
                if key == 27:
                    print('[calib] abort (ESC, abort service, or stdin)', flush=True)
                    raise KeyboardInterrupt
                if key == ord('q'):
                    print('[calib] finish_calibration: running solve...', flush=True)
                    break
                continue

            frame_bgr = host.latest_frame.copy()
            gray = ensure_gray(frame_bgr)

            try:
                charuco_corners, charuco_ids, marker_corners, marker_ids = detect_charuco_compatible(
                    gray, board, aruco_dict, detector
                )
            except Exception as e:
                print(f'[WARN] Detection error: {e}')
                charuco_corners, charuco_ids, marker_corners, marker_ids = None, None, [], []

            num_corners = 0 if charuco_ids is None else len(charuco_ids)
            info_text = [
                f'source=ros2 topic={args.topic}',
                f'corners={num_corners} kept={acc.kept}',
                'c/Enter=capture  q=finish  esc=quit' if use_gui else 'TTY:Enter=capture | svc capture_frame',
            ]
            vis = draw_live_overlay(frame_bgr, marker_corners, marker_ids, charuco_corners, charuco_ids, info_text)
            if use_gui:
                cv2.imshow('charuco_live', vis)
                key = _merge_live_keys(cv2.waitKey(1) & 0xFF)
            else:
                host.publish_preview(vis)
                key = _merge_live_keys(_consume_headless_service_flags(host.flags))

            if key == 27:
                print('[calib] abort (ESC, abort service, or stdin)', flush=True)
                raise KeyboardInterrupt
            elif key == ord('q'):
                print('[calib] finish_calibration: running solve...', flush=True)
                break
            elif key == ord('c'):
                now = time.time()
                if now - last_capture_time < args.capture_cooldown:
                    print('[calib] capture ignored (cooldown)', flush=True)
                    continue
                frame_name = f'ros2_{frame_idx:04d}'
                ok = acc.try_add_frame(frame_bgr, gray, frame_name, marker_corners, marker_ids, charuco_corners, charuco_ids)
                if ok:
                    print(f'[calib] captured {frame_name} (kept={acc.kept})', flush=True)
                    frame_idx += 1
                    last_capture_time = now
                else:
                    _rej = '[calib] capture rejected (corners/size)'
                    if args.debug_dir:
                        _rej += '; see debug_dir'
                    print(_rej, flush=True)

    finally:
        host.shutdown()
        if use_gui:
            cv2.destroyAllWindows()


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            'Calibrate camera using a ChArUco board.\n\n'
            'Input sources:\n'
            '  --source file : read images from disk glob(s)\n'
            '  --source usb  : capture frames from cv2.VideoCapture(camera_id)\n'
            '  --source ros2 : subscribe ROS2 Image topic via cv_bridge\n\n'
            'Board geometry defaults match charuco_board_gen.py:\n'
            '  -w / --width       5  (squares in X)\n'
            '  -H / --height      7  (squares in Y; -h is reserved for --help)\n'
            '  --aruco_dict       DICT_4X4_50\n\n'
            'PHYSICAL SIZES (no defaults — you must measure on the printed board):\n'
            '  --square_length    real side length of one chessboard square\n'
            '  --marker_size      real side length of one black ArUco marker\n'
            '  Use the same unit for both (e.g. mm or meters). Do not pass\n'
            '  charuco_board_gen.py --square_px / --marker_px here; those are\n'
            '  only pixel sizes for the PNG, not millimeters on paper.\n'
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        'images',
        nargs='*',
        help='image glob(s), only used when --source file, e.g. images/*.jpg'
    )

    parser.add_argument(
        '--source',
        type=str,
        default='file',
        choices=['file', 'usb', 'ros2'],
        help='input source type'
    )

    parser.add_argument(
        '-w', '--width',
        type=int,
        default=5,
        help='squares in X (default: 5; same as charuco_board_gen.py --width)'
    )
    parser.add_argument(
        '-H', '--height',
        type=int,
        default=7,
        help='squares in Y (default: 7; same as charuco_board_gen.py --height; use -H, not -h)'
    )
    parser.add_argument(
        '--aruco_dict',
        type=str,
        default='DICT_4X4_50',
        choices=sorted(ARUCO_DICTS.keys()),
        help='ArUco dictionary (default: DICT_4X4_50; same as charuco_board_gen.py)'
    )

    parser.add_argument(
        '--square_length',
        type=float,
        default=None,
        help=(
            'REQUIRED, measured on the physical print: side length of one chessboard square '
            '(same unit as --marker_size, e.g. mm). Not --square_px from charuco_board_gen.py.'
        )
    )
    parser.add_argument(
        '--marker_size',
        type=float,
        default=None,
        help=(
            'REQUIRED, measured on the physical print: side length of one black ArUco marker '
            '(same unit as --square_length). Not --marker_px from charuco_board_gen.py.'
        )
    )

    parser.add_argument(
        '--min_corners',
        type=int,
        default=6,
        help='minimum detected ChArUco corners per frame to keep it'
    )
    parser.add_argument(
        '--debug_images',
        action='store_true',
        help='write detection debug PNGs under --debug_dir (default: off; no debug folder created)',
    )
    parser.add_argument(
        '--debug_dir',
        type=str,
        default=os.path.join('~', 'camera_calibration_params', 'camera_calib_debug'),
        help='directory for debug PNGs when --debug_images is set (default: ~/camera_calibration_params/camera_calib_debug)',
    )
    parser.add_argument(
        '--output_yaml',
        type=str,
        default=os.path.join('~', 'camera_calibration_params', 'camera_config_for_opencv.yml'),
        help='output calibration YAML (OpenCV FileStorage; default: ~/camera_calibration_params/camera_config_for_opencv.yml)',
    )
    parser.add_argument(
        '--output_json',
        type=str,
        default=os.path.join('~', 'camera_calibration_params', 'camera_config.json'),
        help='output calibration JSON (legacy AMR format, same keys as old calibrate node)'
    )
    parser.add_argument(
        '--no_output_json',
        action='store_true',
        help='do not write camera_config JSON'
    )
    parser.add_argument(
        '--output_camera_info_yaml',
        type=str,
        default=os.path.join('~', 'camera_calibration_params', 'camera_config_for_usb_cam_ros.yaml'),
        help=(
            'ROS usb_cam camera_info YAML (camera_calibration format; set params camera_info_url '
            'to file://...'
        ),
    )
    parser.add_argument(
        '--no_output_camera_info_yaml',
        action='store_true',
        help='do not write ROS usb_cam camera_info YAML',
    )
    parser.add_argument(
        '--camera_info_camera_name',
        type=str,
        default='usb_cam_color',
        help='camera_name in ROS camera_info YAML (match usb_cam params camera_name)',
    )
    parser.add_argument(
        '--requested_width',
        type=int,
        default=None,
        help='optional stream width for JSON requested_width (default: calibrated image width)'
    )
    parser.add_argument(
        '--requested_height',
        type=int,
        default=None,
        help='optional stream height for JSON requested_height (default: calibrated image height)'
    )
    parser.add_argument(
        '--requested_fps',
        type=int,
        default=30,
        help='FPS field in JSON requested_fps (default: 30)'
    )
    parser.add_argument(
        '--save_captures_dir',
        type=str,
        default=os.path.join('~', 'camera_calibration_params', 'charuco_captures'),
        help='directory for accepted raw PNG frames in usb/ros2 mode (default: ~/camera_calibration_params/charuco_captures)'
    )
    parser.add_argument(
        '--no_save_captures',
        action='store_true',
        help='do not save accepted raw frames to disk (usb/ros2 only)'
    )

    # USB
    parser.add_argument(
        '--camera_id',
        type=int,
        default=0,
        help='camera index for usb mode (default: 0)'
    )
    parser.add_argument(
        '--usb_backend',
        type=str,
        default='auto',
        choices=['auto', 'v4l2', 'default'],
        help=(
            'VideoCapture API for --source usb: auto uses V4L2 on Linux first (safer on Jetson; '
            'avoids GStreamer segfaults); v4l2 forces V4L2; default uses OpenCV default backend'
        ),
    )

    # ROS2
    parser.add_argument(
        '--topic',
        type=str,
        default='/camera/image_raw',
        help='ROS2 Image topic for ros2 mode'
    )

    parser.add_argument(
        '--capture_cooldown',
        type=float,
        default=0.5,
        help='minimum time gap between captures in live mode'
    )

    parser.add_argument(
        '--preview',
        type=str,
        default='auto',
        choices=['auto', 'gui', 'topic'],
        help=(
            'Live preview for usb/ros2: auto uses OpenCV window if DISPLAY is set, else publishes '
            'CompressedImage and uses Trigger services for capture/finish/abort'
        ),
    )
    parser.add_argument(
        '--preview_topic',
        type=str,
        default='/charuco_calib/preview/compressed',
        help='sensor_msgs/CompressedImage topic when --preview topic or auto without DISPLAY'
    )
    parser.add_argument(
        '--preview_jpeg_quality',
        type=int,
        default=85,
        help='JPEG quality 1-100 for preview topic (default: 85)'
    )
    parser.add_argument(
        '--ros_node_name',
        type=str,
        default='charuco_calib',
        help='ROS node name for headless services (capture_frame, finish_calibration, abort_calibration)'
    )

    return parser


def validate_args(args):
    if args.square_length is None:
        raise ValueError(
            '--square_length is required (physical measurement on the printed board).\n'
            'Use a ruler/caliper on the real square — do not use charuco_board_gen.py --square_px.'
        )

    if args.marker_size is None:
        raise ValueError(
            '--marker_size is required (physical measurement on the printed board).\n'
            'Use a ruler/caliper on the real black marker — do not use charuco_board_gen.py --marker_px.'
        )

    if args.marker_size >= args.square_length:
        raise ValueError('--marker_size must be smaller than --square_length')

    if args.source == 'file' and len(args.images) == 0:
        raise ValueError('For --source file, provide image glob(s), e.g. "images/*.jpg"')

    if not (1 <= args.preview_jpeg_quality <= 100):
        raise ValueError('--preview_jpeg_quality must be between 1 and 100')


def main():
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)

    if args.debug_images:
        args.debug_dir = os.path.expanduser(args.debug_dir)
    else:
        args.debug_dir = None

    args.output_yaml = os.path.expanduser(args.output_yaml)
    if not args.no_output_camera_info_yaml:
        args.output_camera_info_yaml = os.path.expanduser(args.output_camera_info_yaml)

    if args.source in ('usb', 'ros2'):
        if args.no_save_captures:
            args.save_captures_dir = None
        else:
            args.save_captures_dir = os.path.expanduser(args.save_captures_dir)
    else:
        args.save_captures_dir = None

    use_gui = True
    if args.source in ('usb', 'ros2'):
        use_gui = use_gui_preview(args)

    preview_note = ''
    if args.source in ('usb', 'ros2'):
        mode = 'OpenCV window' if use_gui else f"ROS {args.preview_topic} + Trigger services"
        preview_note = f' | preview={args.preview} ({mode})'
    print(
        f'[calibrate_camera_node] ChArUco calibrator. source={args.source}{preview_note}',
        flush=True,
    )

    if args.debug_dir is not None:
        os.makedirs(args.debug_dir, exist_ok=True)
    if args.source in ('usb', 'ros2') and args.save_captures_dir:
        os.makedirs(args.save_captures_dir, exist_ok=True)

    aruco_dict = get_aruco_dict(args.aruco_dict)
    board, allow_charuco_detector = create_charuco_board(
        args.width,
        args.height,
        args.square_length,
        args.marker_size,
        aruco_dict
    )
    detector = create_detector(board, allow_charuco_detector)

    acc = CalibrationAccumulator(
        board=board,
        min_corners=args.min_corners,
        debug_dir=args.debug_dir,
        save_captures_dir=args.save_captures_dir if args.source in ('usb', 'ros2') else None
    )

    try:
        if args.source == 'file':
            run_file_source(args, board, aruco_dict, detector, acc)
        elif args.source == 'usb':
            run_usb_source(args, board, aruco_dict, detector, acc, use_gui)
        elif args.source == 'ros2':
            run_ros2_source(args, board, aruco_dict, detector, acc, use_gui)
        else:
            raise RuntimeError(f'Unknown source: {args.source}')

        rms, camera_matrix, dist_coeffs, _rvecs, _tvecs = calibrate_from_accumulator(acc, board)

        print('\n===== Calibration Result =====')
        print('Source          :', args.source)
        print('Frames used     :', acc.kept)
        print('Image size      :', acc.image_size)
        print('Board squares X :', args.width)
        print('Board squares Y :', args.height)
        print('Square length   :', args.square_length)
        print('Marker size     :', args.marker_size)
        print('Aruco dict      :', args.aruco_dict)
        print('RMS             :', rms)
        print('Camera matrix:\n', camera_matrix)
        print('Dist coeffs:\n', dist_coeffs.ravel())

        save_calibration_yaml(args.output_yaml, acc.image_size, args, rms, camera_matrix, dist_coeffs)
        print('\nSaved calibration to:', os.path.abspath(args.output_yaml))
        if not args.no_output_json:
            json_path = save_calibration_json(
                args.output_json,
                acc.image_size,
                camera_matrix,
                dist_coeffs,
                rms,
                requested_width=args.requested_width,
                requested_height=args.requested_height,
                requested_fps=args.requested_fps,
            )
            print('Saved camera config (JSON) to:', os.path.abspath(json_path))
        if not args.no_output_camera_info_yaml:
            ci_path = save_ros_usb_cam_camera_info_yaml(
                args.output_camera_info_yaml,
                acc.image_size,
                camera_matrix,
                dist_coeffs,
                camera_name=args.camera_info_camera_name,
                rms=float(rms),
            )
            print('Saved ROS usb_cam camera_info YAML to:', os.path.abspath(ci_path))
        if args.debug_dir:
            print('Debug images saved in:', os.path.abspath(args.debug_dir))
        if args.source in ('usb', 'ros2') and args.save_captures_dir:
            print('Accepted raw frames saved in:', os.path.abspath(args.save_captures_dir))

    except KeyboardInterrupt:
        print('\nAborted by user.')
    finally:
        if use_gui:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
