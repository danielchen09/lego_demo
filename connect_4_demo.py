from spot_apriltag_localize import SpotAprilTag
from align_rgb import RGBAligner
from utils import *
from detect_board_state_yolo import detect_board_state
from game import query_ai

import bosdyn.client.lease
import bosdyn.client.util
from bosdyn.api import image_pb2
from bosdyn.client.frame_helpers import BODY_FRAME_NAME, ODOM_FRAME_NAME, get_a_tform_b, math_helpers
from bosdyn.client.image import build_image_request
from bosdyn.client.robot_command import RobotCommandBuilder, blocking_stand, block_until_arm_arrives

import cv2
import time
import numpy as np

from ultralytics import YOLO


class Connect4Demo(SpotAprilTag):
    DEBUG = True

    LEFT_TAG_ID = 1
    RIGHT_TAG_ID = 2

    # board measurements
    N_COLS = 7
    BOARD_TAG_OFFSET = 0.13 # how much to go forward from tag
    BOARD_SIDE_OFFSET = 0.02 # from tag center to blue part of board
    BOARD_WIDTH = 0.76 # total length of blue part
    BOARD_HEIGHT_FROM_TAG = 0.65 # from tag center to top of board

    # robot parameters
    IMAGE_SOURCE = 'hand_color_image'

    # check board state
    GAZE_HEIGHT = 0.25
    GAZE_DISTANCE = 0.7

    # pick chip
    SCAN_START_ARM_OFFSET_TAG = np.array([-0.025, 0., 0.47])
    PRESCAN_ARM_OFFSET_TAG = np.array([-0.175, 0., 0.55])
    SCAN_START_BODY_OFFSET_TAG = np.array([-0.75, 0])
    SCAN_DIST = 0.4
    SCAN_STEP = 0.005
    GRASP_FORWARD_OFFSET = 0.01
    LIFT_METERS = 0.6

    MIN_MAGENTA_PIXELS = 14000
    MIN_BBOX_WIDTH = 240

    MAGENTA_HSV_LOW = [135, 70, 40]
    MAGENTA_HSV_HIGH = [175, 255, 255]

    ROD_SLOPE = 2 / 41

    PICK_TAG = 'left'

    # dropping
    DROP_HOVER_HEIGHT = 0.07 # drop chip height
    ALIGN_KP = 0.003
    ALIGN_THRESHOLD = 15
    ALIGN_ITERS = 20
    DROP_INITIAL_HOVER_HEIGHT = 0.05


    def __init__(self):
        super().__init__()
        self.rgb_aligner = RGBAligner()
        self.yolo_model = YOLO('models/exp-4.pt')
    
    def run(self):
        while True:
            self.compute_references()

            initial_se2 = self.get_global_transform()

            board_state = self.check_board_state()
            print("Board state:")
            for row in board_state:
                print(' '.join(str(cell) for cell in row))

            self.stow_arm()

            drop_col, winner = self.get_drop_col(board_state)
            print('winner: ', winner)

            self.pick_chip()

            self.global_move_se2(initial_se2.x, initial_se2.y, initial_se2.angle)
            self.compute_references()

            self.drop_chip(drop_col)

            self.stow_arm()

            cont = input('continue? (Y/n)')
            if cont.lower() == 'n':
                break
            self.global_move_se2(initial_se2.x, initial_se2.y, initial_se2.angle)
    
    def compute_references(self):
        tag_poses = self.image_to_tag_poses()

        assert self.LEFT_TAG_ID in tag_poses, f"Left tag {self.LEFT_TAG_ID} not found"
        assert self.RIGHT_TAG_ID in tag_poses, f"Right tag {self.RIGHT_TAG_ID} not found"

        self.left_center = tag_poses[self.LEFT_TAG_ID]['center_world']
        self.right_center = tag_poses[self.RIGHT_TAG_ID]['center_world']

        self.right_dir = normalize((self.right_center - self.left_center) * np.array([1, 1, 0]))
        self.forward_dir = normalize(np.cross(-self.right_dir, np.array([0, 0, 1])))

    def check_board_state(self):
        midpoint = (self.left_center + self.right_center) / 2
        midpoint[2] += self.GAZE_HEIGHT

        hand_pos = midpoint - self.forward_dir * self.GAZE_DISTANCE
        hand_dir_quat = quat_look_at(self.forward_dir)

        self.reach_relative_world(position=hand_pos, quaternion=hand_dir_quat, seconds=3)
        self.open_gripper(seconds=1.)

        image = self.get_images(self.IMAGE_SOURCE)[0]
        cv2.imwrite('output/hand_image.png', image)

        return detect_board_state(self.yolo_model, 'output/hand_image.png')

    def get_drop_col(self, board_state):
        return query_ai(board_state)

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

    def pick_chip(self):
        pick_tag_center = self.left_center if self.PICK_TAG == 'left' else self.right_center
        stand_center_xy = pick_tag_center[:2] + self.SCAN_START_BODY_OFFSET_TAG[0] * self.forward_dir[:2] + self.SCAN_START_BODY_OFFSET_TAG[1] * self.right_dir[:2]
        self.global_move_se2(
            stand_center_xy[0],
            stand_center_xy[1],
            -np.arccos(self.forward_dir[0])
        )

        odom_T_scan_quat = quat_from_points(self.left_center, self.right_center, "-z", "-y")


        odom_T_pre_scan_pos = pick_tag_center + transform_vec(self.PRESCAN_ARM_OFFSET_TAG, self.forward_dir, self.right_dir)
        self.move_arm_world(odom_T_pre_scan_pos, odom_T_scan_quat, gripper_open=True, seconds=2.)

        odom_T_scan_pos = pick_tag_center + transform_vec(self.SCAN_START_ARM_OFFSET_TAG, self.forward_dir, self.right_dir)
        self.move_arm_world(odom_T_scan_pos, odom_T_scan_quat, seconds=2.)

        detected_position = self._scan_down_for_magenta(odom_T_scan_pos, odom_T_scan_quat, self.forward_dir)
        assert detected_position is not None, "No detected"

        odom_T_grasp_pos = detected_position + transform_vec(np.array([self.GRASP_FORWARD_OFFSET, 0, 0]), self.forward_dir, self.right_dir)
        self.move_arm_world(odom_T_grasp_pos, odom_T_scan_quat, seconds=1.0)

        self.close_gripper()

        odom_T_lift_pos = odom_T_scan_pos.copy()
        odom_T_lift_pos[2] = pick_tag_center[2] + self.LIFT_METERS
        self.move_arm_world(odom_T_lift_pos, odom_T_scan_quat, seconds=1.0)
        self.stow_arm()
    
    def drop_chip(self, col):
        board_start = self.left_center + self.right_dir * self.BOARD_SIDE_OFFSET
        board_width = np.linalg.norm(self.right_center - self.left_center)
        col_width = board_width / self.N_COLS
        drop_pos = board_start + self.right_dir * (col * col_width + col_width / 2 + 0.02)

        odom_T_drop_pos = drop_pos + transform_vec(
            np.array([self.BOARD_TAG_OFFSET, 0, self.BOARD_HEIGHT_FROM_TAG + self.DROP_HOVER_HEIGHT]),
            self.forward_dir,
            self.right_dir
        )
        odom_T_drop_quat = quat_from_points(self.left_center, self.right_center, "z", "x")
        self.reach_relative_world(odom_T_drop_pos + np.array([0, 0, self.DROP_INITIAL_HOVER_HEIGHT]), odom_T_drop_quat, seconds=2.0)
        self.reach_relative_world(odom_T_drop_pos, odom_T_drop_quat, seconds=0.5)

        for i in range(20):
            diff, display_image = self.rgb_aligner.align_rgb()
            cv2.imwrite(f"output/debug_{i}.png", display_image)
            print(f'aligning...{diff}')
            if abs(diff) < self.ALIGN_THRESHOLD:
                break
            dp = np.array([0, 1, 0]) * np.sign(diff) * self.ALIGN_KP
            self.reach_relative_arm(-dp, seconds=0.2)

        self.open_gripper()

if __name__ == '__main__':
    demo = Connect4Demo()
    demo.start()