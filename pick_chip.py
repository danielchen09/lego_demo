import argparse
import os
import time

import cv2
import numpy as np

import bosdyn.client.lease
import bosdyn.client.util
from bosdyn.api import image_pb2
from bosdyn.client.frame_helpers import BODY_FRAME_NAME, ODOM_FRAME_NAME, get_a_tform_b, math_helpers
from bosdyn.client.image import build_image_request
from bosdyn.client.robot_command import RobotCommandBuilder, blocking_stand, block_until_arm_arrives

from connect_4_drop import DropChipController, normalize
from utils import *

def transform_vec(vec, forward, right):
    return vec[0] * forward + vec[1] * right + vec[2] * np.array([0, 0, 1])

class PickChipController(DropChipController):
    IMAGE_SOURCE = 'hand_color_image'
    SCAN_START_ARM_OFFSET_TAG = np.array([-0.025, -0.02, 0.47])
    PRESCAN_ARM_OFFSET_TAG = np.array([-0.175, 0., 0.55])
    SCAN_START_BODY_OFFSET_TAG = np.array([-0.75, 0])
    SCAN_DIST = 0.4
    SCAN_STEP = 0.005
    GRASP_FORWARD_OFFSET = 0.0
    LIFT_METERS = 0.6

    MIN_MAGENTA_PIXELS = 14000
    MIN_BBOX_WIDTH = 250

    MAGENTA_HSV_LOW = [135, 70, 40]
    MAGENTA_HSV_HIGH = [175, 255, 255]

    ROD_SLOPE = 2 / 41

    PICK_TAG = 'left'

    def _scan_down_for_magenta(self, scan_start, hand_quat, forward_dir):
        steps = int(self.SCAN_DIST / self.SCAN_STEP) + 1
        stable_count = 0

        for i in range(steps):
            position = scan_start + np.array([0.0, 0.0, -i * self.SCAN_STEP]) + i * self.SCAN_STEP / self.SCAN_DIST * forward_dir * self.ROD_SLOPE
            self.move_arm_world(position, hand_quat, seconds=0.05)
            time.sleep(0.1)

            bgr_image = self._get_gripper_bgr_image(self.IMAGE_SOURCE)
            detection = self._detect_magenta_edge(bgr_image)

            print(
                "scan",
                i,
                "z",
                round(float(position[2]), 4),
                "pixels",
                detection["pixels"],
                "bbox",
                detection["bbox"],
                "found",
                detection["found"],
            )

            if detection["found"]:
                stable_count += 1
            else:
                stable_count = 0

            if stable_count >= 2:
                return position

        return None

    def _get_gripper_bgr_image(self, source_name):
        request = build_image_request(
            source_name,
            quality_percent=100,
            pixel_format=image_pb2.Image.PIXEL_FORMAT_RGB_U8,
        )
        image_response = self.image_client.get_image([request])[0]
        image = image_response.shot.image

        if image.format == image_pb2.Image.FORMAT_JPEG:
            encoded = np.frombuffer(image.data, dtype=np.uint8)
            bgr_image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if bgr_image is None:
                raise RuntimeError("Failed to decode JPEG image from gripper camera.")
            return bgr_image

        if image.pixel_format == image_pb2.Image.PIXEL_FORMAT_RGB_U8:
            rgb_image = np.frombuffer(image.data, dtype=np.uint8).reshape(
                image.rows, image.cols, 3
            )
            return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        if image.pixel_format == image_pb2.Image.PIXEL_FORMAT_RGBA_U8:
            rgba_image = np.frombuffer(image.data, dtype=np.uint8).reshape(
                image.rows, image.cols, 4
            )
            return cv2.cvtColor(rgba_image, cv2.COLOR_RGBA2BGR)

        if image.pixel_format == image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U8:
            grey_image = np.frombuffer(image.data, dtype=np.uint8).reshape(image.rows, image.cols)
            return cv2.cvtColor(grey_image, cv2.COLOR_GRAY2BGR)

        raise RuntimeError(f"Unsupported gripper image pixel format: {image.pixel_format}")

    def _detect_magenta_edge(self, bgr_image):
        hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        lower = np.array(
            self.MAGENTA_HSV_LOW,
            dtype=np.uint8,
        )
        upper = np.array(self.MAGENTA_HSV_HIGH, dtype=np.uint8)
        mask = cv2.inRange(hsv_image, lower, upper)

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {"found": False, "pixels": 0, "bbox": None, "mask": mask}

        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        pixels = cv2.countNonZero(mask)
        found = pixels >= self.MIN_MAGENTA_PIXELS and w >= self.MIN_BBOX_WIDTH
        return {"found": found, "pixels": pixels, "bbox": (x, y, w, h), "mask": mask}


    def run(self):

        tag_poses = self.image_to_tag_poses()

        assert self.LEFT_TAG_ID in tag_poses, f"Left tag {self.LEFT_TAG_ID} not found"
        assert self.RIGHT_TAG_ID in tag_poses, f"Right tag {self.RIGHT_TAG_ID} not found"

        left_center = tag_poses[self.LEFT_TAG_ID]['center_world']
        right_center = tag_poses[self.RIGHT_TAG_ID]['center_world']
        right_dir = normalize((right_center - left_center) * np.array([1, 1, 0]))
        forward_dir = normalize(np.cross(-right_dir, np.array([0, 0, 1])))

        pick_tag_center = left_center if self.PICK_TAG == 'left' else right_center

        stand_center_xy = pick_tag_center[:2] + self.SCAN_START_BODY_OFFSET_TAG[0] * forward_dir[:2] + self.SCAN_START_BODY_OFFSET_TAG[1] * right_dir[:2]
        self.global_move_se2(
            stand_center_xy[0],
            stand_center_xy[1],
            -np.arccos(forward_dir[0])
        )

        odom_T_scan_quat = quat_from_points(left_center, right_center, "-z", "-y")


        odom_T_pre_scan_pos = pick_tag_center + transform_vec(self.PRESCAN_ARM_OFFSET_TAG, forward_dir, right_dir)
        self.move_arm_world(odom_T_pre_scan_pos, odom_T_scan_quat, gripper_open=True, seconds=2.)

        odom_T_scan_pos = pick_tag_center + transform_vec(self.SCAN_START_ARM_OFFSET_TAG, forward_dir, right_dir)
        self.move_arm_world(odom_T_scan_pos, odom_T_scan_quat, seconds=2.)

        detected_position = self._scan_down_for_magenta(odom_T_scan_pos, odom_T_scan_quat, forward_dir)
        assert detected_position is not None, "No detected"

        odom_T_grasp_pos = detected_position + transform_vec(np.array([self.GRASP_FORWARD_OFFSET, 0, 0]), forward_dir, right_dir)
        self.move_arm_world(odom_T_grasp_pos, odom_T_scan_quat, seconds=1.0)

        self.close_gripper()

        odom_T_lift_pos = odom_T_scan_pos.copy()
        odom_T_lift_pos[2] = pick_tag_center[2] + self.LIFT_METERS
        self.move_arm_world(odom_T_lift_pos, odom_T_scan_quat, seconds=1.0)

        input('>')
        self.open_gripper()
        self.stow_arm()




if __name__ == '__main__':
    controller = PickChipController()
    controller.start()