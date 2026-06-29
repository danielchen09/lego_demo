from connect_4_drop import DropChipController, normalize
from utils import *

from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient,
                                         block_until_arm_arrives, blocking_stand)
from bosdyn.client.frame_helpers import GRAV_ALIGNED_BODY_FRAME_NAME, ODOM_FRAME_NAME, get_a_tform_b
from bosdyn.api import (arm_command_pb2, geometry_pb2, robot_command_pb2, synchronized_command_pb2,
                        trajectory_pb2)

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import time
import beepy
import os



class BoardDatasetCollection(DropChipController):
    IMAGE_SOURCE = 'hand_color_image'
    GAZE_HEIGHT = 0.25
    GAZE_DISTANCE = 0.7

    def run(self):
        print([src.name for src in self.image_client.list_image_sources()])
        tag_poses = self.image_to_tag_poses()

        assert self.LEFT_TAG_ID in tag_poses, f"Left tag {self.LEFT_TAG_ID} not found"
        assert self.RIGHT_TAG_ID in tag_poses, f"Right tag {self.RIGHT_TAG_ID} not found"

        left_center = tag_poses[self.LEFT_TAG_ID]['center_world']
        right_center = tag_poses[self.RIGHT_TAG_ID]['center_world']

        odom_T_midpoint = (left_center + right_center) / 2
        odom_T_midpoint[2] += self.GAZE_HEIGHT

        right_dir = normalize((right_center - left_center) * np.array([1, 1, 0]))
        forward_dir = normalize(np.cross(-right_dir, np.array([0, 0, 1])))
        hand_pos = odom_T_midpoint - forward_dir * self.GAZE_DISTANCE


        initial = True
        while True:
            face_dir = forward_dir + np.random.uniform(-0.05, 0.05, (3,))
            perterbed_pos = hand_pos + np.random.uniform(-0.03, 0.03, (3,))
            hand_dir_quat = quat_look_at(face_dir)
            self.reach_relative_world(position=perterbed_pos, quaternion=hand_dir_quat, seconds=3 if initial else 0.5)

            self.open_gripper()

            images = self.get_images(self.IMAGE_SOURCE)
            hand_image = images[0]
            cv2.imwrite(f'data/board/{int(time.time())}.png', hand_image)
            cv2.imwrite(f'data/board.png', hand_image)
            initial = False

            beepy.beep(sound=1)
            cont = input(f'continue(n={len(os.listdir("data/board"))})? (Y/n)')
            if cont.lower() == 'n':
                break
        self.stow_arm()



if __name__ == '__main__':
    controller = BoardDatasetCollection()
    controller.start()
