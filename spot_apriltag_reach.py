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


class SpotAprilTag(SpotController):
    TAG_SIZE_METERS = 0.146
    TAG_HOVER_HEIGHT_METERS = 0.20
    HAND_YAW_OFFSET_RADIANS = 0.0

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

        self._camera_to_extrinsics_guess = self.populate_source_dict()

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
            cv2.polylines(image_grey, [np.int32(bbox)], True, (0, 0, 0), 2)
            tag_poses.append((detections[i].pose_t, detections[i].pose_R, bbox))

        #Rotate each image such that it is always displayed upright for debug output.
        image_grey = self.rotate_image(image_grey, source_name)
        cv2.imwrite(f'img_{source_name}.png', image_grey)
        return tag_poses

    def image_to_tag_poses(self):
        """Determine which camera source has a fiducial.
           Return the pose of the first detected fiducial."""
        #Iterate through all five camera sources to check for a fiducial
        for i in range(len(self._source_names)):
            source_name = self._source_names[i]
            img_req = build_image_request(source_name, quality_percent=100,
                                          image_format=image_pb2.Image.FORMAT_RAW)
            image_response = self.image_client.get_image([img_req])
            self._body_tform_camera = get_a_tform_b(image_response[0].shot.transforms_snapshot,
                                                    BODY_FRAME_NAME,
                                                    image_response[0].shot.frame_name_image_sensor)
            self._body_tform_world = get_a_tform_b(image_response[0].shot.transforms_snapshot,
                                                   BODY_FRAME_NAME, VISION_FRAME_NAME)

            # Camera intrinsics for the given source camera.
            self._intrinsics = image_response[0].source.pinhole.intrinsics
            width = image_response[0].shot.image.cols
            height = image_response[0].shot.image.rows

            # detect given fiducial in image and return the bounding box of it
            tag_poses = self.detect_fiducial_in_image(image_response[0].shot.image, (width, height),
                                                      source_name)
            if tag_poses:
                print(f'Found tag for {source_name}')
                return tag_poses, source_name
            else:
                self._tag_not_located = True
                print(f'Failed to find tag for {source_name}')
        return [], None


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

    def compute_fiducial_in_body_frame(self, tvec):
        """Transform the tag position from camera coordinates to world coordinates."""
        fiducial_rt_camera_frame = np.array(
            [float(tvec[0][0]),
             float(tvec[1][0]),
             float(tvec[2][0])])
        return self._body_tform_camera.transform_point(
            fiducial_rt_camera_frame[0], fiducial_rt_camera_frame[1], fiducial_rt_camera_frame[2])
        # fiducial_rt_world = self._body_tform_world.inverse().transform_point(
        #     body_tform_fiducial[0], body_tform_fiducial[1], body_tform_fiducial[2])
        # return fiducial_rt_world

    def compute_tag_pose_in_body_frame(self, tvec, rot_matrix):
        """Transform the tag pose from camera frame into body frame."""
        camera_tform_tag_matrix = np.eye(4)
        camera_tform_tag_matrix[:3, :3] = rot_matrix
        camera_tform_tag_matrix[:3, 3] = [float(tvec[0][0]), float(tvec[1][0]), float(tvec[2][0])]
        camera_tform_tag = math_helpers.SE3Pose.from_matrix(camera_tform_tag_matrix)
        return self._body_tform_camera * camera_tform_tag

    @staticmethod
    def downward_hand_quat_with_yaw(yaw, yaw_offset=0.0):
        """Keep the hand facing downward while rotating around body z to match tag yaw."""
        body_q_tag_yaw = math_helpers.Quat.from_yaw(yaw + yaw_offset)
        tag_yaw_q_downward_hand = math_helpers.Quat.from_pitch(math.pi / 2.0)
        return body_q_tag_yaw * tag_yaw_q_downward_hand

    def bbox_to_image_object_pts(self, bbox):
        """Determine the object points and image points for the bounding box.
           The origin in object coordinates = top left corner of the fiducial.
           Order both points sets following: (TL,TR, BL, BR)"""
        fiducial_height_and_width = 146  #mm
        obj_pts = np.array([[0, 0], [fiducial_height_and_width, 0], [0, fiducial_height_and_width],
                            [fiducial_height_and_width, fiducial_height_and_width]],
                           dtype=np.float32)
        #insert a 0 as the third coordinate (xyz)
        obj_points = np.insert(obj_pts, 2, 0, axis=1)

        #['lb-rb-rt-lt']
        img_pts = np.array([[bbox[3][0], bbox[3][1]], [bbox[2][0], bbox[2][1]],
                            [bbox[0][0], bbox[0][1]], [bbox[1][0], bbox[1][1]]], dtype=np.float32)
        return obj_points, img_pts

    def pixel_coords_to_camera_coords(self, bbox, intrinsics, source_name):
        """Compute transformation of 2d pixel coordinates to 3d camera coordinates."""
        camera = self.make_camera_matrix(intrinsics)
        # Track a triplet of (translation vector, rotation vector, camera source name)
        best_bbox = (None, None, source_name)
        # The best bounding box is considered the closest to the robot body.
        closest_dist = float('inf')
        for i in range(len(bbox)):
            obj_points, img_points = self.bbox_to_image_object_pts(bbox[i])
            if self._camera_to_extrinsics_guess[source_name][0]:
                # initialize the position estimate with the previous extrinsics solution
                # then iteratively solve for new position
                old_rvec, old_tvec = self._camera_to_extrinsics_guess[source_name][1]
                _, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera, np.zeros((5, 1)),
                                             old_rvec, old_tvec, True, cv2.SOLVEPNP_ITERATIVE)
            else:
                # Determine current extrinsic solution for the tag.
                _, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera, np.zeros((5, 1)))

            # Save extrinsics results to help speed up next attempts to locate bounding box in
            # the same camera source.
            self._camera_to_extrinsics_guess[source_name] = (True, (rvec, tvec))

            dist = math.sqrt(float(tvec[0][0])**2 + float(tvec[1][0])**2 +
                             float(tvec[2][0])**2) / 1000.0
            if dist < closest_dist:
                closest_dist = dist
                best_bbox = (tvec, rvec, source_name)

        # Flag indicating if the best april tag been found/located
        self._tag_not_located = best_bbox[0] is None and best_bbox[1] is None
        return best_bbox
    

    @staticmethod
    def make_camera_params(ints):
        """Return dt_apriltags camera params: fx, fy, cx, cy."""
        return (ints.focal_length.x, ints.focal_length.y, ints.principal_point.x,
                ints.principal_point.y)

    @staticmethod
    def make_camera_matrix(ints):
        """Transform the ImageResponse proto intrinsics into a camera matrix."""
        camera_matrix = np.array([[ints.focal_length.x, ints.skew.x, ints.principal_point.x],
                                  [ints.skew.y, ints.focal_length.y, ints.principal_point.y],
                                  [0, 0, 1]])
        return camera_matrix


    def populate_source_dict(self):
        """Fills dictionary of the most recently computed camera extrinsics with the camera source.
           The initial boolean indicates if the extrinsics guess should be used."""
        camera_to_extrinsics_guess = dict()
        for src in self._source_names:
            # Dictionary values: use_extrinsics_guess bool, (rotation vector, translation vector) tuple.
            camera_to_extrinsics_guess[src] = (False, (None, None))
        return camera_to_extrinsics_guess

    def run(self):
        # image_responses = self.image_client.get_image_from_sources([self.config.image_source])
        tag_poses, source_name = self.image_to_tag_poses()
        if tag_poses:
            (tvec, rot_matrix, _) = min(tag_poses, key=lambda tag_pose: np.linalg.norm(tag_pose[0]))
            body_T_tag = self.compute_tag_pose_in_body_frame(tvec, rot_matrix)
            hand_x = body_T_tag.x
            hand_y = body_T_tag.y
            hand_z = body_T_tag.z + self.TAG_HOVER_HEIGHT_METERS
            fiducial_rt_body = geometry_pb2.Vec3(x=hand_x, y=hand_y, z=0.4)
            body_Q_hand = self.downward_hand_quat_with_yaw(
                body_T_tag.rot.to_yaw(), self.HAND_YAW_OFFSET_RADIANS).to_proto()
            body_T_hand = geometry_pb2.SE3Pose(position=fiducial_rt_body, rotation=body_Q_hand)
            
            print(body_T_hand)

            robot_state = self.robot_state_client.get_robot_state()
            odom_T_body = get_a_tform_b(robot_state.kinematic_state.transforms_snapshot,
                                        ODOM_FRAME_NAME, BODY_FRAME_NAME)
            odom_T_hand = odom_T_body * math_helpers.SE3Pose.from_proto(body_T_hand)

            print('odom_T_hand: {}'.format(odom_T_hand))

            # # duration in seconds
            seconds = 5

            # Create the arm command.
            arm_command = RobotCommandBuilder.arm_pose_command(
                odom_T_hand.x, odom_T_hand.y, odom_T_hand.z, odom_T_hand.rot.w, odom_T_hand.rot.x,
                odom_T_hand.rot.y, odom_T_hand.rot.z, ODOM_FRAME_NAME, seconds)

            # Tell the robot's body to follow the arm
            follow_arm_command = RobotCommandBuilder.follow_arm_command()

            # Combine the arm and mobility commands into one synchronized command.
            command = RobotCommandBuilder.build_synchro_command(follow_arm_command, arm_command)

            # Send the request
            move_command_id = self.command_client.robot_command(command)
            self.robot.logger.info('Moving arm to position.')

            block_until_arm_arrives(self.command_client, move_command_id, 10.0)


if __name__ == '__main__':
    spot_apriltag = SpotAprilTag()
    spot_apriltag.start()
