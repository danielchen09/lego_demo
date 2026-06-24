import argparse
import sys
import time

import cv2
import numpy as np
from PIL import Image
from dt_apriltags import Detector
import math

import bosdyn.client
import bosdyn.client.estop
import bosdyn.client.lease
import bosdyn.client.util
from bosdyn.api import estop_pb2, geometry_pb2, image_pb2, manipulation_api_pb2
from bosdyn.client.estop import EstopClient
from bosdyn.client.frame_helpers import GRAV_ALIGNED_BODY_FRAME_NAME, ODOM_FRAME_NAME, get_a_tform_b, VISION_FRAME_NAME, get_vision_tform_body, math_helpers, BODY_FRAME_NAME
from bosdyn.client.image import ImageClient, build_image_request
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient, blocking_stand, block_until_arm_arrives, blocking_command
from bosdyn.client.robot_state import RobotStateClient

from spot_controller import SpotController

# import viser


class SpotAprilTag(SpotController):
    TAG_SIZE_METERS = 0.1

    def __init__(self):
        super(SpotAprilTag, self).__init__()

        self._source_names = [
            src.name for src in self.image_client.list_image_sources() if
            (src.image_type == image_pb2.ImageSource.IMAGE_TYPE_VISUAL and 'depth' not in src.name)
        ]

        self.detector = Detector(families='tag36h11',
            nthreads=1,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0
        )
    
    def detect_fiducial_in_image(self, image, dim, source_name):
        """Detect fiducials within a single image and return their camera-frame poses."""
        image_grey = np.array(
            Image.frombytes('P', (int(dim[0]), int(dim[1])), data=image.data, decoder_name='raw'))

        camera_params = self.make_camera_params(self._intrinsics)
        detections = self.detector.detect(image_grey, 
                                          estimate_tag_pose=True,
                                          camera_params=camera_params,
                                          tag_size=self.TAG_SIZE_METERS)

        tag_poses = []
        for i in range(len(detections)):
            # Draw the bounding box detection in the image.
            bbox = detections[i].corners
            cv2.polylines(image_grey, [np.int32(bbox)], True, (0, 255, 0), 2)
            # tag_poses.append((detections[i].pose_t, detections[i].pose_R, bbox))
            tag_poses.append(detections[i])

        #Rotate each image such that it is always displayed upright for debug output.
        image_grey = self.rotate_image(image_grey, source_name)
        # cv2.imwrite(f'img_{source_name}.png', image_grey)
        return tag_poses

    def image_to_tag_poses(self):
        """Determine which camera source has a fiducial.
           Return the pose of the first detected fiducial."""
        #Iterate through all five camera sources to check for a fiducial
        transforms = self.robot_state_client.get_robot_state().kinematic_state.transforms_snapshot
        ret = {}
        dets = []

        for i in range(len(self._source_names)):
            source_name = self._source_names[i]
            img_req = build_image_request(source_name, quality_percent=100,
                                          image_format=image_pb2.Image.FORMAT_RAW)
            image_response = self.image_client.get_image([img_req])
            cam_to_body_tform = get_a_tform_b(image_response[0].shot.transforms_snapshot,
                                                    BODY_FRAME_NAME,
                                                    image_response[0].shot.frame_name_image_sensor)
            # Camera intrinsics for the given source camera.
            self._intrinsics = image_response[0].source.pinhole.intrinsics
            width = image_response[0].shot.image.cols
            height = image_response[0].shot.image.rows

            # detect given fiducial in image and return the bounding box of it
            tag_poses = self.detect_fiducial_in_image(image_response[0].shot.image, (width, height),
                                                      source_name)

            if tag_poses:
                print(f'Found tag for {source_name}')
                dets.append([{
                    'center': np.array(cam_to_body_tform.transform_point(
                        tp.pose_t[0],
                        tp.pose_t[1],
                        tp.pose_t[2],
                    )).reshape(3,),
                    'id': tp.tag_id,
                    'source': source_name
                } for tp in tag_poses])
        for det in dets:
            for dd in det:
                ret[dd['id']] = dd
        return ret
    
    def run(self):
        # image_responses = self.image_client.get_image_from_sources([self.config.image_source])
        tag_poses = self.image_to_tag_poses()

        tag_id = int(input(f"id({tag_poses.keys()}): "))

        if tag_id in tag_poses:
            body_tform_goal = tag_poses[tag_id]['center'] + np.array([0, 0, 0.8])
            cur_se2 = self.get_global_transform()
            print(cur_se2)
            self.reach_relative_body(body_tform_goal, np.array([1, 0, 0, 0]))
            input('continue>')
            self.stow_arm()
            self.global_move_se2(cur_se2.x, cur_se2.y, cur_se2.angle)


        
        print(tag_poses)


if __name__ == '__main__':
    spot_apriltag = SpotAprilTag()
    spot_apriltag.start()
