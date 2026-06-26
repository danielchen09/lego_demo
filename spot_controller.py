import argparse
import sys
import time
import os

import cv2
import numpy as np

import bosdyn.client
import bosdyn.client.estop
import bosdyn.client.lease
import bosdyn.client.util
from bosdyn.api import estop_pb2, geometry_pb2, image_pb2, manipulation_api_pb2
from bosdyn.client.estop import EstopClient
from bosdyn.client.frame_helpers import GRAV_ALIGNED_BODY_FRAME_NAME, ODOM_FRAME_NAME, get_a_tform_b, VISION_FRAME_NAME, get_vision_tform_body, math_helpers, get_se2_a_tform_b, BODY_FRAME_NAME, HAND_FRAME_NAME
from bosdyn.client.image import ImageClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient, blocking_stand, block_until_arm_arrives, blocking_command, blocking_sit, block_for_trajectory_cmd
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.world_object import WorldObjectClient
from bosdyn.api.basic_command_pb2 import RobotCommandFeedbackStatus


class SpotController:
    def __init__(self):
        os.environ['BOSDYN_CLIENT_USERNAME'] = 'admin'
        os.environ['BOSDYN_CLIENT_PASSWORD'] = '6ayqcfmx6io3'

        self.config = self.get_args()
        config = self.config
        bosdyn.client.util.setup_logging(config.verbose)

        sdk = bosdyn.client.create_standard_sdk('ArmObjectGraspClient')
        self.robot = sdk.create_robot(config.hostname)
        bosdyn.client.util.authenticate(self.robot)
        self.robot.time_sync.wait_for_sync()

        self._verify_estop()

        self.lease_client = self.robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)
        self.robot_state_client: RobotStateClient = self.robot.ensure_client(RobotStateClient.default_service_name)
        self.image_client: ImageClient = self.robot.ensure_client(ImageClient.default_service_name)
        self.manipulation_api_client: ManipulationApiClient = self.robot.ensure_client(ManipulationApiClient.default_service_name)
        self._world_object_client: WorldObjectClient = self.robot.ensure_client(WorldObjectClient.default_service_name)
        self.command_client: RobotCommandClient = self.robot.ensure_client(RobotCommandClient.default_service_name)
        self.robot_state_client: RobotStateClient = self.robot.ensure_client(RobotStateClient.default_service_name)

    def get_images(self, *source_list):
        image_responses = self.image_client.get_image_from_sources(source_list)
        images = []
        for image in image_responses:
            if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
                dtype = np.uint16
            else:
                dtype = np.uint8
            img = np.fromstring(image.shot.image.data, dtype=dtype)
            if image.shot.image.format == image_pb2.Image.FORMAT_RAW:
                img = img.reshape(image.shot.image.rows, image.shot.image.cols)
            else:
                img = cv2.imdecode(img, -1)
            images.append(img)
        return images

    def get_args(self):
        parser = argparse.ArgumentParser()
        bosdyn.client.util.add_base_arguments(parser)
        return parser.parse_args()

    def _verify_estop(self):
        """Verify the robot is not estopped"""

        client = self.robot.ensure_client(EstopClient.default_service_name)
        if client.get_status().stop_level != estop_pb2.ESTOP_LEVEL_NONE:
            error_message = 'Robot is estopped. Please use an external E-Stop client, such as the' \
                            ' estop SDK example, to configure E-Stop.'
            self.robot.logger.error(error_message)
            raise Exception(error_message)

    def start(self):
        with bosdyn.client.lease.LeaseKeepAlive(self.lease_client, must_acquire=True, return_at_exit=True):
            self.robot.logger.info('Powering on robot... This may take a several seconds.')
            self.robot.power_on(timeout_sec=20)
            assert self.robot.is_powered_on(), 'Robot power on failed.'
            self.robot.logger.info('Robot powered on.')

            self.robot.logger.info('Commanding robot to stand...')
            
            blocking_stand(self.command_client, timeout_sec=10)
            self.robot.logger.info('Robot standing.')
            
            # unstow = RobotCommandBuilder.arm_ready_command()

            # unstow_command_id = command_client.robot_command(unstow)
            # self.robot.logger.info('Unstow command issued.')

            # block_until_arm_arrives(command_client, unstow_command_id, 3.0)

            cur_se2 = self.get_global_transform()
            try:
                self.run()
            finally:
                new_se2 = self.get_global_transform()
                self.global_move_se2(cur_se2.x, cur_se2.y, new_se2.angle)

            self.shutdown()

    def shutdown(self):
        self.robot.logger.info('Sitting down and turning off.')

        blocking_sit(self.command_client)
        
        self.robot.logger.info('Powering off')

        # Power the robot off. By specifying "cut_immediately=False", a safe power off command
        # is issued to the robot. This will attempt to sit the robot before powering off.
        self.robot.power_off(cut_immediately=False, timeout_sec=20)
        assert not self.robot.is_powered_on(), 'Robot power off failed.'
        self.robot.logger.info('Robot safely powered off.')
    
    def run(self):
        pass

    def stow_arm(self):
        stow = RobotCommandBuilder.arm_stow_command()

        # Issue the command via the RobotCommandClient
        stow_command_id = self.command_client.robot_command(stow)

        block_until_arm_arrives(self.command_client, stow_command_id, 3.0)

    def reach_relative_body(self, position: np.ndarray, quaternion: np.ndarray, seconds=5):

        body_tform_hand = geometry_pb2.SE3Pose(
            position=geometry_pb2.Vec3(x=position[0], y=position[1], z=position[2]),
            rotation=geometry_pb2.Quaternion(w=quaternion[0], x=quaternion[1], y=quaternion[2], z=quaternion[3])
        )
        transforms = self.get_transforms_snapshot()
        odom_tform_body = get_a_tform_b(transforms, ODOM_FRAME_NAME, BODY_FRAME_NAME)
        odom_tform_hand = odom_tform_body * math_helpers.SE3Pose.from_proto(body_tform_hand)
        self.reach_relative_world(
            (odom_tform_hand.x, odom_tform_hand.y, odom_tform_hand.z),
            (odom_tform_hand.rot.w, odom_tform_hand.rot.x, odom_tform_hand.rot.y, odom_tform_hand.rot.z),
            seconds
        )
    
    def reach_relative_arm(self, position, seconds=5):
        transforms = self.get_transforms_snapshot()
        odom_tform_hand = get_a_tform_b(transforms, ODOM_FRAME_NAME, HAND_FRAME_NAME)
        hand_tform_hand = geometry_pb2.SE3Pose(
            position=geometry_pb2.Vec3(x=position[0], y=position[1], z=position[2]),
            rotation=geometry_pb2.Quaternion(w=1, x=0, y=0, z=0)
        )
        tf = odom_tform_hand * math_helpers.SE3Pose.from_proto(hand_tform_hand)
        self.reach_relative_world(
            (tf.x, tf.y, tf.z),
            (tf.rot.w, tf.rot.x, tf.rot.y, tf.rot.z),
            seconds
        )

    def reach_relative_world(self, position: np.ndarray, quaternion: np.ndarray, seconds=5):
        arm_command = RobotCommandBuilder.arm_pose_command(
            *position,
            *quaternion,
            ODOM_FRAME_NAME,
            seconds
        )
        follow_arm_command = RobotCommandBuilder.follow_arm_command()
        command = RobotCommandBuilder.build_synchro_command(
            follow_arm_command,
            arm_command
        )
        reach_command_id = self.command_client.robot_command(command)
        block_until_arm_arrives(self.command_client, reach_command_id, 10.0)


    def get_transforms_snapshot(self):
        return self.robot_state_client.get_robot_state().kinematic_state.transforms_snapshot

    def get_global_transform(self):
        transforms = self.get_transforms_snapshot()
        body_tfrom_ident = math_helpers.SE2Pose(0, 0, 0)
        world_tform_body = get_se2_a_tform_b(transforms, ODOM_FRAME_NAME, BODY_FRAME_NAME)
        return world_tform_body * body_tfrom_ident

    def global_move_se2(self, x, y, yaw):
        robot_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
            goal_x=x,
            goal_y=y,
            goal_heading=yaw,
            frame_name=ODOM_FRAME_NAME,
            params=RobotCommandBuilder.mobility_params(stair_hint=False)
        )
        command_id = self.command_client.robot_command(lease=None, command=robot_cmd,
                                                end_time_secs=time.time() + 10.0)
        while True:
            feedback = self.command_client.robot_command_feedback(command_id)
            mobility_feedback = feedback.feedback.synchronized_feedback.mobility_command_feedback
            if mobility_feedback.status != RobotCommandFeedbackStatus.STATUS_PROCESSING:
                print('Failed to reach the goal')
                return False
            traj_feedback = mobility_feedback.se2_trajectory_feedback
            if (traj_feedback.status == traj_feedback.STATUS_AT_GOAL and
                    traj_feedback.body_movement_status == traj_feedback.BODY_STATUS_SETTLED):
                print('Arrived at the goal.')
                return True
            time.sleep(1)

    def open_gripper(self, return_cmd=False):
        gripper_command = RobotCommandBuilder.claw_gripper_open_command()
        cmd_id = self.command_client.robot_command(gripper_command)
        block_until_arm_arrives(self.command_client, cmd_id, 4.0)
    
    def close_gripper(self):
        gripper_command = RobotCommandBuilder.claw_gripper_close_command()
        cmd_id = self.command_client.robot_command(gripper_command)
        block_until_arm_arrives(self.command_client, cmd_id, 4.0)

    def arm_look_at(self, position_world, gripper_open=True, seconds=4.0, hand_pose=None):
        gaze_command = RobotCommandBuilder.arm_gaze_command(
            position_world[0],
            position_world[1],
            position_world[2],
            ODOM_FRAME_NAME,
            frame2_tform_desired_hand=hand_pose,
            frame2_name=None if hand_pose is None else ODOM_FRAME_NAME
        )
        command = RobotCommandBuilder.claw_gripper_open_command()
        if gripper_open:
            command = RobotCommandBuilder.build_synchro_command(command, gaze_command)
        gaze_command_id = self.command_client.robot_command(command)

        return block_until_arm_arrives(self.command_client, gaze_command_id, seconds)

    def move_arm_world(self, position_world, quat_world, seconds=5, gripper_open=False):
        arm_command = RobotCommandBuilder.arm_pose_command(
            *position_world,
            *quat_world,
            ODOM_FRAME_NAME,
            seconds
        )
        if gripper_open:
            gripper_command = RobotCommandBuilder.claw_gripper_open_command()
            arm_command = RobotCommandBuilder.build_synchro_command(arm_command, gripper_command)
        command_id = self.command_client.robot_command(arm_command)
        block_until_arm_arrives(self.command_client, command_id, seconds + 5.0)

    @staticmethod
    def rotate_image(image, source_name):
        """Rotate the image so that it is always displayed upright."""
        if source_name == 'frontleft_fisheye_image':
            image = cv2.rotate(image, rotateCode=0)
        elif source_name == 'right_fisheye_image':
            image = cv2.rotate(image, rotateCode=1)
        elif source_name == 'frontright_fisheye_image':
            image = cv2.rotate(image, rotateCode=0)
        return image
    

    @staticmethod
    def make_camera_params(ints):
        """Return dt_apriltags camera params: fx, fy, cx, cy."""
        return (ints.focal_length.x, ints.focal_length.y, ints.principal_point.x,
                ints.principal_point.y)