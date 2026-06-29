#!/usr/bin/env python3
"""Return magenta alignment error from RealSense D405 RGB frames."""

import json
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


WIDTH = 640
HEIGHT = 480
FPS = 30

MAGENTA_HUE_LOW = 135
MAGENTA_HUE_HIGH = 175
MIN_SATURATION = 70
MIN_VALUE = 40
IGNORE_HALF_WIDTH = 2
DEFAULT_CENTER_LINE_CONFIG = (
    Path(__file__).resolve().parent / "configs" / "cam_center_line.json"
)


def normalize_center_line(line):
    """Orient the line top-to-bottom so side labels are click-order independent."""
    line = np.array(line, dtype=np.float32)
    if (line[1, 1], line[1, 0]) < (line[0, 1], line[0, 0]):
        line = line[[1, 0]]
    return line


def load_center_line(config_path=DEFAULT_CENTER_LINE_CONFIG):
    """Load two image points defining the center line, or return None."""
    config_path = Path(config_path)
    if not config_path.exists():
        return None

    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    points = data.get("points")
    if not isinstance(points, list) or len(points) != 2:
        raise ValueError(f"{config_path} must contain exactly two line points")

    line = np.array(points, dtype=np.float32)
    if line.shape != (2, 2):
        raise ValueError(f"{config_path} line points must be [[x1, y1], [x2, y2]]")

    if np.linalg.norm(line[1] - line[0]) == 0:
        raise ValueError(f"{config_path} line points must not be identical")

    return normalize_center_line(line)


def default_center_line(width, height):
    center_x = width // 2
    return normalize_center_line([[center_x, 0], [center_x, height - 1]])


def line_signed_distances(shape, line):
    height, width = shape
    p1, p2 = line
    dx, dy = p2 - p1
    length = np.hypot(dx, dy)
    if length == 0:
        raise ValueError("Center line points must not be identical")

    yy, xx = np.indices((height, width), dtype=np.float32)
    signed_area = dx * (yy - p1[1]) - dy * (xx - p1[0])
    return signed_area / length


def clip_line_to_image(line, width, height):
    p1, p2 = line.astype(np.float32)
    direction = p2 - p1
    candidates = []

    if direction[0] != 0:
        for x in (0, width - 1):
            t = (x - p1[0]) / direction[0]
            y = p1[1] + t * direction[1]
            if 0 <= y <= height - 1:
                candidates.append((int(round(x)), int(round(y))))

    if direction[1] != 0:
        for y in (0, height - 1):
            t = (y - p1[1]) / direction[1]
            x = p1[0] + t * direction[0]
            if 0 <= x <= width - 1:
                candidates.append((int(round(x)), int(round(y))))

    unique = []
    for point in candidates:
        if point not in unique:
            unique.append(point)

    if len(unique) >= 2:
        return unique[0], unique[1]

    return tuple(line[0].astype(int)), tuple(line[1].astype(int))


class RGBAligner:
    """Measure magenta-pixel imbalance around a calibrated RGB image line."""

    def __init__(
        self,
        width=WIDTH,
        height=HEIGHT,
        fps=FPS,
        magenta_hue_low=MAGENTA_HUE_LOW,
        magenta_hue_high=MAGENTA_HUE_HIGH,
        min_saturation=MIN_SATURATION,
        min_value=MIN_VALUE,
        ignore_half_width=IGNORE_HALF_WIDTH,
        line_config_path=DEFAULT_CENTER_LINE_CONFIG,
    ):
        self.width = width
        self.height = height
        self.fps = fps
        self.magenta_hue_low = magenta_hue_low
        self.magenta_hue_high = magenta_hue_high
        self.min_saturation = min_saturation
        self.min_value = min_value
        self.ignore_half_width = ignore_half_width
        self.line_config_path = Path(line_config_path)
        self.pipeline = rs.pipeline()
        self.started = False
        self.center_line = self._load_center_line()

    def _load_center_line(self):
        line = load_center_line(self.line_config_path)
        if line is None:
            return default_center_line(self.width, self.height)
        return line

    def start(self):
        if self.started:
            return

        config = rs.config()
        config.enable_stream(
            rs.stream.color,
            self.width,
            self.height,
            rs.format.rgb8,
            self.fps,
        )
        self.pipeline.start(config)
        self.started = True

    def stop(self):
        if not self.started:
            return

        self.pipeline.stop()
        self.started = False

    def _build_magenta_mask(self, hsv_image):
        lower = np.array(
            [self.magenta_hue_low, self.min_saturation, self.min_value],
            dtype=np.uint8,
        )
        upper = np.array([self.magenta_hue_high, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv_image, lower, upper)

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def align_rgb(self):
        """Return right magenta pixel count minus left magenta pixel count."""
        self.start()

        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return 0

        rgb_image = np.asanyarray(color_frame.get_data())
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        magenta_mask = self._build_magenta_mask(hsv_image)

        height, width = magenta_mask.shape
        ignore_half_width = max(0, self.ignore_half_width)
        signed_distances = line_signed_distances(magenta_mask.shape, self.center_line)
        left_count = cv2.countNonZero(
            np.where(signed_distances > ignore_half_width, magenta_mask, 0)
        )
        right_count = cv2.countNonZero(
            np.where(signed_distances < -ignore_half_width, magenta_mask, 0)
        )

        display_image = bgr_image.copy()
        contours, _ = cv2.findContours(
            magenta_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(display_image, contours, -1, (0, 255, 255), 1)
        line_start, line_end = clip_line_to_image(self.center_line, width, height)
        cv2.line(display_image, line_start, line_end, (0, 255, 0), 2)

        p1, p2 = self.center_line
        direction = p2 - p1
        line_length = np.linalg.norm(direction)
        normal = np.array([direction[1], -direction[0]], dtype=np.float32) / line_length
        for offset in (-ignore_half_width, ignore_half_width):
            boundary = self.center_line + normal * offset
            boundary_start, boundary_end = clip_line_to_image(boundary, width, height)
            cv2.line(display_image, boundary_start, boundary_end, (0, 255, 0), 1)

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

        return right_count - left_count, display_image

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        self.stop()


def align_rgb():
    """One-shot helper that returns right magenta pixels minus left pixels."""
    aligner = RGBAligner()
    try:
        return aligner.align_rgb()
    finally:
        aligner.stop()


if __name__ == "__main__":
    aligner = RGBAligner()
    while True:
        error, display_image = aligner.align_rgb()
        print(f"(right - left): {error}")
        cv2.imshow("RGB Alignment", display_image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    aligner.stop()
    cv2.destroyAllWindows()
