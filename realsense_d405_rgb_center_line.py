#!/usr/bin/env python3
"""Display the RealSense D405 RGB feed with magenta highlighting."""

import argparse

import cv2
import numpy as np
import pyrealsense2 as rs


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Show RealSense D405 RGB frames, highlight magenta pixels, and show "
            "the HSV value under the cursor."
        )
    )
    parser.add_argument("--width", type=int, default=640, help="Color frame width.")
    parser.add_argument("--height", type=int, default=480, help="Color frame height.")
    parser.add_argument("--fps", type=int, default=30, help="Color stream FPS.")
    parser.add_argument(
        "--magenta-hue-low",
        type=int,
        default=135,
        help="Lower OpenCV HSV hue threshold for magenta, 0-179.",
    )
    parser.add_argument(
        "--magenta-hue-high",
        type=int,
        default=175,
        help="Upper OpenCV HSV hue threshold for magenta, 0-179.",
    )
    parser.add_argument(
        "--min-saturation",
        type=int,
        default=70,
        help="Minimum HSV saturation for magenta detection, 0-255.",
    )
    parser.add_argument(
        "--min-value",
        type=int,
        default=40,
        help="Minimum HSV value for magenta detection, 0-255.",
    )
    parser.add_argument(
        "--background-saturation-scale",
        type=float,
        default=0.25,
        help="Saturation multiplier for non-magenta pixels.",
    )
    parser.add_argument(
        "--ignore-half-width",
        type=int,
        default=5,
        help=(
            "Pixels to ignore on each side of the center line when counting. "
            "The two boundary lines are drawn at center +/- this value."
        ),
    )
    parser.add_argument(
        "--serial",
        help="Optional RealSense serial number to select a specific camera.",
    )
    return parser.parse_args()


def on_mouse(event, x, y, _flags, cursor):
    if event == cv2.EVENT_MOUSEMOVE:
        cursor["x"] = x
        cursor["y"] = y


def build_magenta_mask(hsv_image, args):
    lower = np.array(
        [args.magenta_hue_low, args.min_saturation, args.min_value],
        dtype=np.uint8,
    )
    upper = np.array([args.magenta_hue_high, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv_image, lower, upper)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def main():
    args = parse_args()

    pipeline = rs.pipeline()
    config = rs.config()

    if args.serial:
        config.enable_device(args.serial)

    config.enable_stream(
        rs.stream.color,
        args.width,
        args.height,
        rs.format.rgb8,
        args.fps,
    )

    pipeline.start(config)
    window_name = "RealSense D405 RGB"
    cursor = {"x": None, "y": None}
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse, cursor)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            rgb_image = np.asanyarray(color_frame.get_data())
            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
            magenta_mask = build_magenta_mask(hsv_image, args)

            muted_hsv = hsv_image.copy()
            muted_hsv[:, :, 1] = np.clip(
                muted_hsv[:, :, 1].astype(np.float32)
                * args.background_saturation_scale,
                0,
                255,
            ).astype(np.uint8)

            display_image = cv2.cvtColor(muted_hsv, cv2.COLOR_HSV2BGR)
            display_image[magenta_mask > 0] = bgr_image[magenta_mask > 0]

            contours, _ = cv2.findContours(
                magenta_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(display_image, contours, -1, (0, 255, 255), 1)

            center_x = display_image.shape[1] // 2
            image_width = display_image.shape[1]
            ignore_half_width = max(0, args.ignore_half_width)
            left_boundary_x = max(0, center_x - ignore_half_width)
            right_boundary_x = min(image_width - 1, center_x + ignore_half_width)
            left_count = cv2.countNonZero(magenta_mask[:, :left_boundary_x])
            right_count = cv2.countNonZero(magenta_mask[:, right_boundary_x + 1 :])

            cv2.line(
                display_image,
                (center_x, 0),
                (center_x, display_image.shape[0] - 1),
                (0, 255, 0),
                2,
            )
            cv2.line(
                display_image,
                (left_boundary_x, 0),
                (left_boundary_x, display_image.shape[0] - 1),
                (0, 255, 0),
                1,
            )
            cv2.line(
                display_image,
                (right_boundary_x, 0),
                (right_boundary_x, display_image.shape[0] - 1),
                (0, 255, 0),
                1,
            )

            count_label = f"L:R {left_count}:{right_count}"
            cv2.putText(
                display_image,
                count_label,
                (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 0),
                4,
                cv2.LINE_AA,
            )
            cv2.putText(
                display_image,
                count_label,
                (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            if cursor["x"] is not None and cursor["y"] is not None:
                x = min(max(cursor["x"], 0), hsv_image.shape[1] - 1)
                y = min(max(cursor["y"], 0), hsv_image.shape[0] - 1)
                hue, saturation, value = hsv_image[y, x]
                red, green, blue = rgb_image[y, x]
                label = (
                    f"x={x} y={y} HSV=({hue},{saturation},{value}) "
                    f"RGB=({red},{green},{blue})"
                )
                cv2.circle(display_image, (x, y), 4, (255, 255, 255), 1)
                cv2.putText(
                    display_image,
                    label,
                    (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 0),
                    4,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    display_image,
                    label,
                    (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            cv2.imshow(window_name, display_image)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
