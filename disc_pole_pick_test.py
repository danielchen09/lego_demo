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

from spot_apriltag_localize import SpotAprilTag


class DiscPolePickTester(SpotAprilTag):
    HOLDER_TAG_ID = 1

    def __init__(self):
        super().__init__()
        self._source_names = [
            name
            for name in self._source_names
            if "hand" not in name and "gripper" not in name
        ]
        if self.config.apriltag_image_source:
            self._source_names = [self.config.apriltag_image_source]
        print(f"Using AprilTag image sources: {self._source_names}")

    def get_args(self):
        parser = argparse.ArgumentParser()
        bosdyn.client.util.add_base_arguments(parser)
        parser.add_argument("--holder-tag-id", type=int, default=self.HOLDER_TAG_ID)
        parser.add_argument(
            "--apriltag-image-source",
            default="frontright_fisheye_image",
            help="Optional fixed Spot body camera source for holder tag detection.",
        )
        parser.add_argument("--gripper-image-source", default=None)
        parser.add_argument("--output-dir", default="output/disc_pole_pick")

        parser.add_argument("--desired-tag-x-meters", type=float, default=0.75)
        parser.add_argument("--desired-tag-y-meters", type=float, default=0.0)
        parser.add_argument("--max-base-move-meters", type=float, default=0.8)
        parser.add_argument("--base-align-iterations", type=int, default=3)
        parser.add_argument("--base-align-tolerance-meters", type=float, default=0.035)
        parser.add_argument("--skip-base-align", action="store_true")

        parser.add_argument("--scan-up-meters", type=float, default=0.47)
        parser.add_argument("--scan-forward-meters", type=float, default=0.025)
        parser.add_argument("--scan-left-meters", type=float, default=0.0)
        parser.add_argument("--pre-scan-backoff-meters", type=float, default=0.15)
        parser.add_argument("--pre-scan-up-extra-meters", type=float, default=0.08)
        parser.add_argument("--scan-distance-meters", type=float, default=0.20)
        parser.add_argument("--scan-step-meters", type=float, default=0.01)
        parser.add_argument("--scan-move-seconds", type=float, default=0.3)

        parser.add_argument(
            "--hand-quat",
            nargs=4,
            type=float,
            default=[1.0, 0.0, 0.0, 0.0],
            metavar=("W", "X", "Y", "Z"),
            help="Body-frame hand quaternion. Tune this for the sideways gripper pose.",
        )
        parser.add_argument(
            "--grasp-twist-deg",
            type=float,
            default=90.0,
            help="Extra wrist twist before grasping.",
        )
        parser.add_argument(
            "--grasp-twist-axis",
            choices=["x", "y", "z"],
            default="x",
            help="Local hand axis for --grasp-twist-deg.",
        )

        parser.add_argument("--magenta-hue-low", type=int, default=135)
        parser.add_argument("--magenta-hue-high", type=int, default=175)
        parser.add_argument("--min-saturation", type=int, default=70)
        parser.add_argument("--min-value", type=int, default=40)
        parser.add_argument("--min-magenta-pixels", type=int, default=14000)
        parser.add_argument("--min-bbox-width-px", type=int, default=250)
        parser.add_argument("--stable-frames", type=int, default=2)

        parser.add_argument("--grasp-forward-meters", type=float, default=0.035)
        parser.add_argument("--lift-meters", type=float, default=0.20)
        return parser.parse_args()

    def start(self):
        self._home_se2 = None
        with bosdyn.client.lease.LeaseKeepAlive(
            self.lease_client, must_acquire=True, return_at_exit=True
        ):
            self.robot.logger.info("Powering on robot... This may take a several seconds.")
            self.robot.power_on(timeout_sec=20)
            assert self.robot.is_powered_on(), "Robot power on failed."
            self.robot.logger.info("Robot powered on.")

            self.robot.logger.info("Commanding robot to stand...")
            blocking_stand(self.command_client, timeout_sec=10)
            self.robot.logger.info("Robot standing.")

            self._home_se2 = self.get_global_transform()

            try:
                self.run()
            except KeyboardInterrupt:
                print("\nCtrl-C received. Returning to home before shutdown.")
            finally:
                self._return_home_and_shutdown()

    def run(self):
        self._create_output_run_dir()

        source_name = self._select_gripper_image_source()
        print(f"Using gripper image source: {source_name}")

        tag_center = self._get_holder_tag_center("before base align")
        print(f"Holder tag center body frame before base align: {tag_center}")

        if not self.config.skip_base_align:
            tag_center = self._align_base_to_holder_tag(tag_center)

        scan_start = tag_center + np.array(
            [
                self.config.scan_forward_meters,
                self.config.scan_left_meters,
                self.config.scan_up_meters,
            ]
        )
        hand_quat = np.array(self.config.hand_quat, dtype=float)
        scan_quat = self._twist_quat(
            hand_quat, self.config.grasp_twist_axis, self.config.grasp_twist_deg
        )

        print(f"Holder tag center body frame after base align: {tag_center}")
        print(f"Scan start body frame: {scan_start}")

        self.open_gripper()
        pre_scan_position = scan_start + np.array(
            [
                -self.config.pre_scan_backoff_meters,
                0.0,
                self.config.pre_scan_up_extra_meters,
            ]
        )
        print(f"Pre-scan body frame: {pre_scan_position}")
        self._reach_body_arm_only(pre_scan_position, scan_quat, seconds=2.0)
        self._reach_body_arm_only(scan_start, scan_quat, seconds=2.0)

        detected_position = self._scan_down_for_magenta(source_name, scan_start, scan_quat)
        if detected_position is None:
            print("No disc edge detected during scan.")
            return

        print(f"Detected likely top disc edge at body-frame pose: {detected_position}")
        answer = input("Move closer, close gripper, and lift? [y/N] ").strip().lower()
        if answer != "y":
            return

        grasp_position = detected_position + np.array([self.config.grasp_forward_meters, 0.0, 0.0])
        lift_position = grasp_position + np.array([0.0, 0.0, self.config.lift_meters])

        self._reach_body_arm_only(detected_position, scan_quat, seconds=1.0)
        self._reach_body_arm_only(grasp_position, scan_quat, seconds=1.0)
        self.close_gripper()
        self._reach_body_arm_only(lift_position, scan_quat, seconds=1.5)

        answer = input("Release disc, stow, and return home? [y/N] ").strip().lower()
        if answer == "y":
            self.open_gripper()
        self.stow_arm()

    def _create_output_run_dir(self):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.output_run_dir = os.path.join(self.config.output_dir, timestamp)
        os.makedirs(self.output_run_dir, exist_ok=True)
        print(f"Saving gripper scan images to: {self.output_run_dir}")

    def _get_holder_tag_center(self, label):
        tag_poses = self.image_to_tag_poses()
        if self.config.holder_tag_id not in tag_poses:
            raise RuntimeError(
                f"Holder tag {self.config.holder_tag_id} not found {label}. "
                f"Detected tags: {list(tag_poses.keys())}"
            )
        return tag_poses[self.config.holder_tag_id]["center"]

    def _align_base_to_holder_tag(self, tag_center):
        for i in range(self.config.base_align_iterations):
            error = np.array(
                [
                    tag_center[0] - self.config.desired_tag_x_meters,
                    tag_center[1] - self.config.desired_tag_y_meters,
                ],
                dtype=float,
            )
            error_norm = np.linalg.norm(error)
            print(
                f"Base align iteration {i + 1}: "
                f"tag_xy=({tag_center[0]:.3f}, {tag_center[1]:.3f}) "
                f"error=({error[0]:.3f}, {error[1]:.3f})"
            )
            if error_norm <= self.config.base_align_tolerance_meters:
                print("Base is within requested tag standoff tolerance.")
                return tag_center

            self._move_base_to_tag_standoff(tag_center)
            tag_center = self._get_holder_tag_center("after base align")

        return tag_center

    def _return_home_and_shutdown(self):
        try:
            if self.robot.is_powered_on():
                try:
                    self.stow_arm()
                except Exception as exc:
                    print(f"Could not stow arm during cleanup: {exc}")

                if self._home_se2 is not None:
                    print("Returning base to home pose.")
                    try:
                        self.global_move_se2(
                            self._home_se2.x,
                            self._home_se2.y,
                            self._home_se2.angle,
                        )
                    except Exception as exc:
                        print(f"Could not return to home pose: {exc}")

                self.shutdown()
        except Exception as exc:
            print(f"Cleanup failed: {exc}")

    def _move_base_to_tag_standoff(self, tag_center):
        body_dx = float(tag_center[0] - self.config.desired_tag_x_meters)
        body_dy = float(tag_center[1] - self.config.desired_tag_y_meters)
        move_norm = np.linalg.norm([body_dx, body_dy])

        if move_norm > self.config.max_base_move_meters:
            scale = self.config.max_base_move_meters / move_norm
            body_dx *= scale
            body_dy *= scale
            print(
                "Base align move clipped to",
                round(self.config.max_base_move_meters, 3),
                "meters.",
            )

        if abs(body_dx) < 0.03 and abs(body_dy) < 0.03:
            print("Base is already close to requested tag standoff.")
            return

        current = self.get_global_transform()
        cos_yaw = np.cos(current.angle)
        sin_yaw = np.sin(current.angle)
        odom_dx = cos_yaw * body_dx - sin_yaw * body_dy
        odom_dy = sin_yaw * body_dx + cos_yaw * body_dy

        goal_x = current.x + odom_dx
        goal_y = current.y + odom_dy
        goal_yaw = current.angle

        print(
            "Aligning base to tag standoff:",
            f"body_delta=({body_dx:.3f}, {body_dy:.3f})",
            f"odom_goal=({goal_x:.3f}, {goal_y:.3f}, {goal_yaw:.3f})",
        )
        self.global_move_se2(goal_x, goal_y, goal_yaw)

    def _reach_body_arm_only(self, position, quaternion, seconds=5):
        body_tform_hand = math_helpers.SE3Pose(
            x=float(position[0]),
            y=float(position[1]),
            z=float(position[2]),
            rot=math_helpers.Quat(
                w=float(quaternion[0]),
                x=float(quaternion[1]),
                y=float(quaternion[2]),
                z=float(quaternion[3]),
            ),
        )
        transforms = self.get_transforms_snapshot()
        odom_tform_body = get_a_tform_b(transforms, ODOM_FRAME_NAME, BODY_FRAME_NAME)
        odom_tform_hand = odom_tform_body * body_tform_hand

        arm_command = RobotCommandBuilder.arm_pose_command(
            odom_tform_hand.x,
            odom_tform_hand.y,
            odom_tform_hand.z,
            odom_tform_hand.rot.w,
            odom_tform_hand.rot.x,
            odom_tform_hand.rot.y,
            odom_tform_hand.rot.z,
            ODOM_FRAME_NAME,
            seconds,
        )
        command_id = self.command_client.robot_command(arm_command)
        block_until_arm_arrives(self.command_client, command_id, seconds + 5.0)

    @staticmethod
    def _twist_quat(quaternion, axis, degrees):
        base = math_helpers.Quat(
            w=float(quaternion[0]),
            x=float(quaternion[1]),
            y=float(quaternion[2]),
            z=float(quaternion[3]),
        )
        half_angle = np.deg2rad(degrees) / 2.0
        xyz = {"x": [1.0, 0.0, 0.0], "y": [0.0, 1.0, 0.0], "z": [0.0, 0.0, 1.0]}[axis]
        twist = math_helpers.Quat(
            w=float(np.cos(half_angle)),
            x=float(np.sin(half_angle) * xyz[0]),
            y=float(np.sin(half_angle) * xyz[1]),
            z=float(np.sin(half_angle) * xyz[2]),
        )
        result = base * twist
        return np.array([result.w, result.x, result.y, result.z])

    def _select_gripper_image_source(self):
        sources = self.image_client.list_image_sources()
        visual_sources = [
            src.name
            for src in sources
            if src.image_type == image_pb2.ImageSource.IMAGE_TYPE_VISUAL
        ]
        print(f"Available visual image sources: {visual_sources}")

        if self.config.gripper_image_source:
            return self.config.gripper_image_source

        preferred_names = [
            "hand_color_image",
            "hand_image",
            "hand_color",
            "gripper_color_image",
            "gripper_image",
        ]
        for name in preferred_names:
            if name in visual_sources:
                return name

        hand_matches = [name for name in visual_sources if "hand" in name or "gripper" in name]
        if hand_matches:
            return hand_matches[0]

        raise RuntimeError(
            "Could not auto-select a gripper camera source. "
            "Run with --gripper-image-source using one of the printed sources."
        )

    def _scan_down_for_magenta(self, source_name, scan_start, hand_quat):
        steps = int(self.config.scan_distance_meters / self.config.scan_step_meters) + 1
        stable_count = 0

        for i in range(steps):
            position = scan_start + np.array([0.0, 0.0, -i * self.config.scan_step_meters])
            self._reach_body_arm_only(position, hand_quat, seconds=self.config.scan_move_seconds)
            time.sleep(0.1)

            bgr_image = self._get_gripper_bgr_image(source_name)
            detection = self._detect_magenta_edge(bgr_image)
            self._write_debug_image(i, bgr_image, detection)

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

            if stable_count >= self.config.stable_frames:
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
            [self.config.magenta_hue_low, self.config.min_saturation, self.config.min_value],
            dtype=np.uint8,
        )
        upper = np.array([self.config.magenta_hue_high, 255, 255], dtype=np.uint8)
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
        found = pixels >= self.config.min_magenta_pixels and w >= self.config.min_bbox_width_px
        return {"found": found, "pixels": pixels, "bbox": (x, y, w, h), "mask": mask}

    def _write_debug_image(self, index, bgr_image, detection):
        display = bgr_image.copy()
        bbox = detection["bbox"]
        if bbox is not None:
            x, y, w, h = bbox
            color = (0, 255, 0) if detection["found"] else (0, 255, 255)
            cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)

        label = f"pixels={detection['pixels']} found={detection['found']}"
        cv2.putText(
            display,
            label,
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            display,
            label,
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.imwrite(os.path.join(self.output_run_dir, f"raw_{index:03d}.png"), bgr_image)
        cv2.imwrite(os.path.join(self.output_run_dir, f"scan_{index:03d}.png"), display)
        cv2.imwrite(os.path.join(self.output_run_dir, f"mask_{index:03d}.png"), detection["mask"])


if __name__ == "__main__":
    controller = DiscPolePickTester()
    controller.start()
