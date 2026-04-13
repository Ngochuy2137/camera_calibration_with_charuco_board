#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import cv2


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

# Optional dictionaries on some newer builds
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
    # Newer OpenCV Python bindings
    if hasattr(cv2.aruco, 'CharucoBoard'):
        return cv2.aruco.CharucoBoard(
            (squares_x, squares_y),
            square_len,
            marker_len,
            aruco_dict
        )

    # Older OpenCV Python bindings
    if hasattr(cv2.aruco, 'CharucoBoard_create'):
        return cv2.aruco.CharucoBoard_create(
            squares_x,
            squares_y,
            square_len,
            marker_len,
            aruco_dict
        )

    raise RuntimeError('Your cv2.aruco does not support CharucoBoard.')


def render_board_image(board, out_size, margin, border_bits):
    # Newer API
    if hasattr(board, 'generateImage'):
        return board.generateImage(out_size, marginSize=margin, borderBits=border_bits)

    # Older API
    if hasattr(board, 'draw'):
        return board.draw(out_size, marginSize=margin, borderBits=border_bits)

    raise RuntimeError('This board object does not support generateImage/draw.')


def main():
    parser = argparse.ArgumentParser(
        description=(
            'Generate a ChArUco board image for printing.\n\n'
            'Defaults for -w/-H/--aruco_dict match calibrate_camera_node.py board geometry.\n'
            '--square_px and --marker_px are only for the PNG resolution, not real-world units;\n'
            'after printing, measure the board and pass those sizes to calibrate_camera_node.py.'
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        '-w', '--width',
        type=int,
        default=5,
        help='squares in X (default: 5; same default as calibrate_camera_node.py)',
    )
    parser.add_argument(
        '-H', '--height',
        type=int,
        default=7,
        help='squares in Y (default: 7; same default as calibrate_camera_node.py; use -H, not -h)',
    )
    parser.add_argument(
        '--square_px',
        type=int,
        default=200,
        help='square size in output image pixels only (not mm on paper; calibrate uses measured size)',
    )
    parser.add_argument(
        '--marker_px',
        type=int,
        default=120,
        help='marker size in output image pixels only (not mm on paper; calibrate uses measured size)',
    )
    parser.add_argument('--margin_px', type=int, default=20, help='margin in output image pixels')
    parser.add_argument(
        '--aruco_dict',
        type=str,
        default='DICT_4X4_50',
        choices=sorted(ARUCO_DICTS.keys()),
        help='ArUco dictionary (default: DICT_4X4_50; same default as calibrate_camera_node.py)',
    )
    parser.add_argument('-o', '--output', type=str, default='charuco_board.png', help='output PNG path')
    args = parser.parse_args()

    if args.marker_px >= args.square_px:
        raise ValueError('--marker_px must be smaller than --square_px')

    aruco_dict = get_aruco_dict(args.aruco_dict)
    board = create_charuco_board(
        args.width,
        args.height,
        float(args.square_px),
        float(args.marker_px),
        aruco_dict
    )

    img_w = args.width * args.square_px + 2 * args.margin_px
    img_h = args.height * args.square_px + 2 * args.margin_px

    img = render_board_image(board, (img_w, img_h), args.margin_px, 1)

    ok = cv2.imwrite(args.output, img)
    if not ok:
        raise RuntimeError(f'Failed to save image: {args.output}')

    print('Saved board image to:', os.path.abspath(args.output))
    print('Print at 100% scale / Actual size.')
    print('After printing, measure the REAL square and marker sizes on paper.')
    print('Use those measured values in calibration, not these pixel values.')


if __name__ == '__main__':
    main()
