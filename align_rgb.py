#!/usr/bin/env python3
"""Return magenta alignment error from RealSense D405 RGB frames."""

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
IGNORE_HALF_WIDTH = 5


class RGBAligner:
    """Measure magenta-pixel imbalance around the RGB image center."""

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
    ):
        self.width = width
        self.height = height
        self.fps = fps
        self.magenta_hue_low = magenta_hue_low
        self.magenta_hue_high = magenta_hue_high
        self.min_saturation = min_saturation
        self.min_value = min_value
        self.ignore_half_width = ignore_half_width
        self.pipeline = rs.pipeline()
        self.started = False

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

        center_x = magenta_mask.shape[1] // 2
        ignore_half_width = max(0, self.ignore_half_width)
        left_boundary_x = max(0, center_x - ignore_half_width)
        right_boundary_x = min(magenta_mask.shape[1] - 1, center_x + ignore_half_width)

        left_count = cv2.countNonZero(magenta_mask[:, :left_boundary_x])
        right_count = cv2.countNonZero(magenta_mask[:, right_boundary_x + 1 :])

        display_image = bgr_image.copy()
        contours, _ = cv2.findContours(
            magenta_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(display_image, contours, -1, (0, 255, 255), 1)
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
    print(align_rgb())
