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
from bosdyn.client.frame_helpers import GRAV_ALIGNED_BODY_FRAME_NAME, ODOM_FRAME_NAME, get_a_tform_b, VISION_FRAME_NAME, get_vision_tform_body, math_helpers
from bosdyn.client.image import ImageClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient, blocking_stand, block_until_arm_arrives, blocking_command, blocking_sit
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.world_object import WorldObjectClient


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
        self.robot_state_client = self.robot.ensure_client(RobotStateClient.default_service_name)
        self.image_client = self.robot.ensure_client(ImageClient.default_service_name)
        self.manipulation_api_client = self.robot.ensure_client(ManipulationApiClient.default_service_name)
        self._world_object_client = self.robot.ensure_client(WorldObjectClient.default_service_name)
        self.command_client = self.robot.ensure_client(RobotCommandClient.default_service_name)
        self.robot_state_client = self.robot.ensure_client(RobotStateClient.default_service_name)


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

            self.run()

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