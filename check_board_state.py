from connect_4_drop import DropChipController, normalize

from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient,
                                         block_until_arm_arrives, blocking_stand)
from bosdyn.client.frame_helpers import GRAV_ALIGNED_BODY_FRAME_NAME, ODOM_FRAME_NAME, get_a_tform_b
from bosdyn.api import (arm_command_pb2, geometry_pb2, robot_command_pb2, synchronized_command_pb2,
                        trajectory_pb2)

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from ultralytics import YOLO

from detect_board_state_yolo import detect_board_state


def quat_look_at(dir_odom: np.ndarray):
    x_axis = dir_odom / np.linalg.norm(dir_odom)

    world_up = np.array([0.0, 0.0, 1.0])

    # If looking almost straight up/down, pick another up reference.
    if abs(np.dot(x_axis, world_up)) > 0.98:
        world_up = np.array([0.0, 1.0, 0.0])

    y_axis = np.cross(world_up, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    z_axis = np.cross(x_axis, y_axis)

    # Columns are the desired local hand axes expressed in odom.
    rot_mat = np.column_stack([x_axis, y_axis, z_axis])

    # scipy gives [x, y, z, w], Spot wants w, x, y, z.
    qx, qy, qz, qw = R.from_matrix(rot_mat).as_quat()
    return qw, qx, qy, qz

class CheckBoardStateController(DropChipController):
    IMAGE_SOURCE = 'hand_color_image'
    GAZE_HEIGHT = 0.25
    GAZE_DISTANCE = 0.7

    def __init__(self):
        super().__init__()
        self.yolo_model = YOLO('models/exp-4.pt')

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
        hand_dir_quat = quat_look_at(forward_dir)



        odom_T_hand = geometry_pb2.SE3Pose(
            position=geometry_pb2.Vec3(x=hand_pos[0], y=hand_pos[1], z=hand_pos[2]),
            rotation=geometry_pb2.Quaternion(w=hand_dir_quat[0], x=hand_dir_quat[1], y=hand_dir_quat[2], z=hand_dir_quat[3]),
        )

        self.reach_relative_world(position=hand_pos, quaternion=hand_dir_quat, seconds=3)


        # self.arm_look_at(odom_T_midpoint, hand_pose=odom_T_hand)

        self.open_gripper()


        images = self.get_images(self.IMAGE_SOURCE, 'hand_depth_in_hand_color_frame')
        hand_image = images[0]
        cv2.imwrite('output/hand_image.png', hand_image)
        board = detect_board_state(self.yolo_model, 'output/hand_image.png')

        for row in board:
            print(' '.join(map(str, row)))

        self.stow_arm()


if __name__ == '__main__':
    controller = CheckBoardStateController()
    controller.start()
