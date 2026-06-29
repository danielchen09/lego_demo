#!/usr/bin/env python3
"""Calibrate the RGB center line from two clicked RealSense image points."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "cam_center_line.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Click two RGB image points to define the calibrated center line."
    )
    parser.add_argument("--width", type=int, default=640, help="Color frame width.")
    parser.add_argument("--height", type=int, default=480, help="Color frame height.")
    parser.add_argument("--fps", type=int, default=30, help="Color stream FPS.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="JSON file to save the calibrated line.",
    )
    parser.add_argument(
        "--serial",
        help="Optional RealSense serial number to select a specific camera.",
    )
    return parser.parse_args()


def load_existing_points(config_path):
    if not config_path.exists():
        return []

    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    points = data.get("points", [])
    if len(points) != 2:
        return []

    return [(int(point[0]), int(point[1])) for point in points]


def on_mouse(event, x, y, _flags, state):
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    if len(state["points"]) >= 2:
        state["points"].clear()

    state["points"].append((x, y))


def draw_points_and_line(image, points):
    for index, point in enumerate(points, start=1):
        cv2.circle(image, point, 5, (0, 255, 255), -1)
        cv2.putText(
            image,
            str(index),
            (point[0] + 8, point[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            str(index),
            (point[0] + 8, point[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    if len(points) == 2:
        cv2.line(image, points[0], points[1], (0, 255, 0), 2)


def draw_status(image, text):
    cv2.putText(
        image,
        text,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def save_points(config_path, points, width, height):
    config_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "image_width": width,
        "image_height": height,
        "points": [[int(x), int(y)] for x, y in points],
    }
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def main():
    args = parse_args()
    state = {
        "points": load_existing_points(args.config),
        "status": f"Click 2 points, s saves to {args.config}, q quits",
    }

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
    window_name = "Calibrate Camera Center Line"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse, state)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            rgb_image = np.asanyarray(color_frame.get_data())
            display_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            draw_points_and_line(display_image, state["points"])
            draw_status(display_image, state["status"])

            cv2.imshow(window_name, display_image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                if len(state["points"]) == 2:
                    save_points(args.config, state["points"], args.width, args.height)
                    state["status"] = f"Saved {args.config}"
                    print(state["status"])
                else:
                    state["status"] = "Click exactly 2 points before saving"
                    print(state["status"])
            elif key in (ord("q"), 27):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
