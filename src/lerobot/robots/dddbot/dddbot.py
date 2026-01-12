#!/usr/bin/env python

import logging
import sys
import time
from functools import cached_property
from typing import Any

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import (
    FeetechMotorsBus,
    OperatingMode,
)
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_dddbot import DddBotConfig

logger = logging.getLogger(__name__)


class DddBot(Robot):
    """
    DddBot: Dual Arm + 4-Wheel Mecanum Mobile Base
    Left Bus: Left Arm (7 DoF) + Base (4 Wheels)
    Right Bus: Right Arm (7 DoF)
    """

    config_class = DddBotConfig
    name = "dddbot"

    def __init__(self, config: DddBotConfig):
        super().__init__(config)
        self.config = config
        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100

        # Left Bus: Left Arm + Base
        self.left_bus = FeetechMotorsBus(
            port=self.config.left_port,
            motors={
                # Left Arm
                "arm_left_shoulder_pan": Motor(1, "sts3215", norm_mode_body),
                "arm_left_shoulder_lift": Motor(2, "sts3215", norm_mode_body),
                "arm_left_shoulder_rotate": Motor(3, "sts3215", norm_mode_body),
                "arm_left_elbow_flex": Motor(4, "sts3215", norm_mode_body),
                "arm_left_wrist_flex": Motor(5, "sts3215", norm_mode_body),
                "arm_left_wrist_roll": Motor(6, "sts3215", norm_mode_body),
                "arm_left_gripper": Motor(7, "sts3215", MotorNormMode.RANGE_0_100),
                # 4-Wheel Mecanum Base
                "base_fl": Motor(21, "sts3215", MotorNormMode.RANGE_M100_100),  # Front Left
                "base_fr": Motor(22, "sts3215", MotorNormMode.RANGE_M100_100),  # Front Right
                "base_rl": Motor(23, "sts3215", MotorNormMode.RANGE_M100_100),  # Rear Left
                "base_rr": Motor(24, "sts3215", MotorNormMode.RANGE_M100_100),  # Rear Right
            },
            calibration=self.calibration,
        )

        # Right Bus: Right Arm
        self.right_bus = FeetechMotorsBus(
            port=self.config.right_port,
            motors={
                # Right Arm
                "arm_right_shoulder_pan": Motor(1, "sts3215", norm_mode_body),
                "arm_right_shoulder_lift": Motor(2, "sts3215", norm_mode_body),
                "arm_right_shoulder_rotate": Motor(3, "sts3215", norm_mode_body),
                "arm_right_elbow_flex": Motor(4, "sts3215", norm_mode_body),
                "arm_right_wrist_flex": Motor(5, "sts3215", norm_mode_body),
                "arm_right_wrist_roll": Motor(6, "sts3215", norm_mode_body),
                "arm_right_gripper": Motor(7, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
        )

        self.left_arm_motors = [m for m in self.left_bus.motors if m.startswith("arm_left_")]
        self.base_motors = [m for m in self.left_bus.motors if m.startswith("base_")]
        self.right_arm_motors = [m for m in self.right_bus.motors if m.startswith("arm_right_")]

        self.cameras = make_cameras_from_configs(config.cameras)

        # Base Geometry
        self.wheel_radius = config.wheel_radius
        self.lx = config.lx
        self.ly = config.ly

    @property
    def _state_ft(self) -> dict[str, type]:
        """Define state features: Dual Arm positions + Base velocity"""
        return dict.fromkeys(
            (
                # Left Arm
                "arm_left_shoulder_pan.pos",
                "arm_left_shoulder_lift.pos",
                "arm_left_shoulder_rotate.pos",
                "arm_left_elbow_flex.pos",
                "arm_left_wrist_flex.pos",
                "arm_left_wrist_roll.pos",
                "arm_left_gripper.pos",
                # Right Arm
                "arm_right_shoulder_pan.pos",
                "arm_right_shoulder_lift.pos",
                "arm_right_shoulder_rotate.pos",
                "arm_right_elbow_flex.pos",
                "arm_right_wrist_flex.pos",
                "arm_right_wrist_roll.pos",
                "arm_right_gripper.pos",
                # Base
                "x.vel",
                "y.vel",
                "theta.vel",
            ),
            float,
        )

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._state_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._state_ft

    @property
    def is_connected(self) -> bool:
        cams_ok = all(cam.is_connected for cam in self.cameras.values())
        return self.left_bus.is_connected and self.right_bus.is_connected and cams_ok

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.left_bus.connect()
        self.right_bus.connect()
        if not self.is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
            )
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.left_bus.is_calibrated and self.right_bus.is_calibrated

    def calibrate(self) -> None:
        """
        Dual Arm + Base Calibration:
        - Left Arm: Position Mode -> Half-turn Homing -> Record ROM
        - Base: No homing (continuous), Fixed ROM 0-4095
        - Right Arm: Position Mode -> Half-turn Homing -> Record ROM
        """
        if self.calibration:
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, "
                f"or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info("Writing existing calibration to both buses")

                calib_left = {k: v for k, v in self.calibration.items() if k in self.left_bus.motors}
                self.left_bus.write_calibration(calib_left, cache=False)
                self.left_bus.calibration = calib_left

                calib_right = {k: v for k, v in self.calibration.items() if k in self.right_bus.motors}
                self.right_bus.write_calibration(calib_right, cache=False)
                self.right_bus.calibration = calib_right

                return

        logger.info(f"\nRunning calibration of {self} (dual-arm + mecanum base)")

        # === Left Arm Calibration ===
        self.left_bus.disable_torque(self.left_arm_motors)
        for name in self.left_arm_motors:
            self.left_bus.write("Operating_Mode", name, OperatingMode.POSITION.value)

        input("Move LEFT arm to the middle of its range of motion, then press ENTER...")
        left_homing = self.left_bus.set_half_turn_homings(self.left_arm_motors)

        # Base wheels are full turn
        for wheel in self.base_motors:
            left_homing[wheel] = 0

        motors_left_all = self.left_arm_motors + self.base_motors
        full_turn_left = [m for m in motors_left_all if m.startswith("base_")]
        unknown_left = [m for m in motors_left_all if m not in full_turn_left]

        print("Move LEFT arm joints sequentially through full ROM. Press ENTER to stop...")
        l_mins, l_maxs = self.left_bus.record_ranges_of_motion(unknown_left)
        for m in full_turn_left:
            l_mins[m] = 0
            l_maxs[m] = 4095

        # === Right Arm Calibration ===
        self.right_bus.disable_torque(self.right_arm_motors)
        for name in self.right_arm_motors:
            self.right_bus.write("Operating_Mode", name, OperatingMode.POSITION.value)

        input("Move RIGHT arm to the middle of its range of motion, then press ENTER...")
        right_homing = self.right_bus.set_half_turn_homings(self.right_arm_motors)

        print("Move RIGHT arm joints sequentially through full ROM. Press ENTER to stop...")
        r_mins, r_maxs = self.right_bus.record_ranges_of_motion(self.right_arm_motors)

        # === Merge Calibration Data ===
        self.calibration = {}

        for name, motor in self.left_bus.motors.items():
            self.calibration[name] = MotorCalibration(
                id=motor.id,
                drive_mode=0,
                homing_offset=left_homing.get(name, 0),
                range_min=l_mins.get(name, 0),
                range_max=l_maxs.get(name, 4095),
            )

        for name, motor in self.right_bus.motors.items():
            self.calibration[name] = MotorCalibration(
                id=motor.id,
                drive_mode=0,
                homing_offset=right_homing.get(name, 0),
                range_min=r_mins.get(name, 0),
                range_max=r_maxs.get(name, 4095),
            )

        # Write back calibration
        calib_left = {k: v for k, v in self.calibration.items() if k in self.left_bus.motors}
        self.left_bus.write_calibration(calib_left, cache=False)
        self.left_bus.calibration = calib_left

        calib_right = {k: v for k, v in self.calibration.items() if k in self.right_bus.motors}
        self.right_bus.write_calibration(calib_right, cache=False)
        self.right_bus.calibration = calib_right

        self._save_calibration()
        print("Calibration saved to", self.calibration_fpath)

    def configure(self):
        """Configure Motor Parameters"""
        # Left Arm: Position Mode
        self.left_bus.disable_torque()
        self.left_bus.configure_motors()
        for name in self.left_arm_motors:
            self.left_bus.write("Operating_Mode", name, OperatingMode.POSITION.value)
            self.left_bus.write("P_Coefficient", name, 16)
            self.left_bus.write("I_Coefficient", name, 0)
            self.left_bus.write("D_Coefficient", name, 32)

        # Base: Velocity Mode
        for name in self.base_motors:
            self.left_bus.write("Operating_Mode", name, OperatingMode.VELOCITY.value)

        # Right Arm: Position Mode
        self.right_bus.disable_torque()
        self.right_bus.configure_motors()
        for name in self.right_arm_motors:
            self.right_bus.write("Operating_Mode", name, OperatingMode.POSITION.value)
            self.right_bus.write("P_Coefficient", name, 16)
            self.right_bus.write("I_Coefficient", name, 0)
            self.right_bus.write("D_Coefficient", name, 32)

    # ==================== 4-Wheel Mecanum Kinematics ====================

    @staticmethod
    def _degps_to_raw(degps: float) -> int:
        """Convert angular velocity (deg/s) to raw value"""
        steps_per_deg = 4096.0 / 360.0
        speed_in_steps = degps * steps_per_deg
        speed_int = int(round(speed_in_steps))
        # Cap the value to fit within signed 16-bit range (-32768 to 32767)
        if speed_int > 0x7FFF:
            speed_int = 0x7FFF
        elif speed_int < -0x8000:
            speed_int = -0x8000
        return speed_int

    @staticmethod
    def _raw_to_degps(raw_speed: int) -> float:
        """Convert raw value to angular velocity (deg/s)"""
        steps_per_deg = 4096.0 / 360.0
        magnitude = raw_speed
        degps = magnitude / steps_per_deg
        return degps

    def _inverse_kinematics(self, vx: float, vy: float, omega: float) -> tuple[float, float, float, float]:
        """
        Inverse Kinematics: Body Velocity -> Wheel Linear Velocity
        
        Args:
            vx: Forward velocity (m/s)
            vy: Lateral velocity (m/s)
            omega: Rotational velocity (rad/s)
            
        Returns:
            (v_fl, v_fr, v_rl, v_rr): Wheel linear velocities (m/s)
        """
        K = self.lx + self.ly
        
        v_fl = vx - vy - K * omega  # Front Left
        v_fr = vx + vy + K * omega  # Front Right
        v_rl = vx + vy - K * omega  # Rear Left
        v_rr = vx - vy + K * omega  # Rear Right
        
        return v_fl, v_fr, v_rl, v_rr

    def _forward_kinematics(
        self, v_fl: float, v_fr: float, v_rl: float, v_rr: float
    ) -> tuple[float, float, float]:
        """
        Forward Kinematics: Wheel Linear Velocity -> Body Velocity
        """
        K = self.lx + self.ly
        
        vx = (v_fl + v_fr + v_rl + v_rr) / 4.0
        vy = (-v_fl + v_fr + v_rl - v_rr) / 4.0
        omega = (-v_fl + v_fr - v_rl + v_rr) / (4.0 * K)
        
        return vx, vy, omega

    def _body_to_wheel_raw(
        self,
        x: float,
        y: float,
        theta: float,
        max_raw: int = 3000,
    ) -> dict:
        """
        Convert body frame velocity to wheel raw commands
        """
        # Convert rotation from deg/s to rad/s
        theta_rad = theta * (np.pi / 180.0)
        
        # Inverse Kinematics
        v_fl, v_fr, v_rl, v_rr = self._inverse_kinematics(x, y, theta_rad)
        
        # Apply Motor Direction Correction
        dirs = self.config.motor_directions
        v_fl *= dirs["base_fl"]
        v_fr *= dirs["base_fr"]
        v_rl *= dirs["base_rl"]
        v_rr *= dirs["base_rr"]
        
        # Linear Speed -> Angular Speed (rad/s)
        w_fl = v_fl / self.wheel_radius
        w_fr = v_fr / self.wheel_radius
        w_rl = v_rl / self.wheel_radius
        w_rr = v_rr / self.wheel_radius
        
        # Convert to deg/s
        wheel_degps = np.array([w_fl, w_fr, w_rl, w_rr]) * (180.0 / np.pi)

        # Scale to avoid exceeding max ref
        steps_per_deg = 4096.0 / 360.0
        raw_floats = [abs(degps) * steps_per_deg for degps in wheel_degps]
        max_raw_computed = max(raw_floats)
        if max_raw_computed > max_raw:
            scale = max_raw / max_raw_computed
            wheel_degps = wheel_degps * scale

        # Convert to raw int
        wheel_raw = [self._degps_to_raw(deg) for deg in wheel_degps]

        return {
            "base_fl": wheel_raw[0],
            "base_fr": wheel_raw[1],
            "base_rl": wheel_raw[2],
            "base_rr": wheel_raw[3],
        }

    def _wheel_raw_to_body(
        self,
        fl_speed: int,
        fr_speed: int,
        rl_speed: int,
        rr_speed: int,
    ) -> dict[str, Any]:
        """
        Convert wheel raw feedback to body frame velocity
        """
        wheel_degps = np.array(
            [
                self._raw_to_degps(fl_speed),
                self._raw_to_degps(fr_speed),
                self._raw_to_degps(rl_speed),
                self._raw_to_degps(rr_speed),
            ]
        )

        wheel_radps = wheel_degps * (np.pi / 180.0)
        wheel_linear_speeds = wheel_radps * self.wheel_radius

        # Apply Motor Direction Correction
        dirs = self.config.motor_directions
        wheel_linear_speeds[0] *= dirs["base_fl"]
        wheel_linear_speeds[1] *= dirs["base_fr"]
        wheel_linear_speeds[2] *= dirs["base_rl"]
        wheel_linear_speeds[3] *= dirs["base_rr"]

        # Forward Kinematics
        vx, vy, omega_rad = self._forward_kinematics(
            wheel_linear_speeds[0],
            wheel_linear_speeds[1],
            wheel_linear_speeds[2],
            wheel_linear_speeds[3],
        )

        theta = omega_rad * (180.0 / np.pi)
        return {
            "x.vel": vx,
            "y.vel": vy,
            "theta.vel": theta,
        }

    # ==================== Observation & Action ====================

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start = time.perf_counter()

        # Read Left Bus (Arm + Base)
        left_pos = self.left_bus.sync_read("Present_Position", self.left_arm_motors)
        base_wheel_vel = self.left_bus.sync_read("Present_Velocity", self.base_motors)
        base_vel = self._wheel_raw_to_body(
            base_wheel_vel["base_fl"],
            base_wheel_vel["base_fr"],
            base_wheel_vel["base_rl"],
            base_wheel_vel["base_rr"],
        )
        # Read Right Bus (Arm)
        right_pos = self.right_bus.sync_read("Present_Position", self.right_arm_motors)

        left_arm_state = {f"{k}.pos": v for k, v in left_pos.items()}
        right_arm_state = {f"{k}.pos": v for k, v in right_pos.items()}
        obs_dict = {**left_arm_state, **right_arm_state, **base_vel}

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # Current Protection
        self.read_and_check_currents(limit_ma=2000, print_currents=False)

        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        left_pos = {k: v for k, v in action.items() if k.endswith(".pos") and k.startswith("arm_left_")}
        right_pos = {k: v for k, v in action.items() if k.endswith(".pos") and k.startswith("arm_right_")}
        base_goal_vel = {k: v for k, v in action.items() if k.endswith(".vel")}

        base_wheel_goal_vel = self._body_to_wheel_raw(
            base_goal_vel.get("x.vel", 0.0),
            base_goal_vel.get("y.vel", 0.0),
            base_goal_vel.get("theta.vel", 0.0),
        )

        if left_pos and self.config.max_relative_target is not None:
            present_left = self.left_bus.sync_read("Present_Position", self.left_arm_motors)
            gp_left = {k: (v, present_left[k.replace(".pos", "")]) for k, v in left_pos.items()}
            left_pos = ensure_safe_goal_position(gp_left, self.config.max_relative_target)

        if right_pos and self.config.max_relative_target is not None:
            present_right = self.right_bus.sync_read("Present_Position", self.right_arm_motors)
            gp_right = {k: (v, present_right[k.replace(".pos", "")]) for k, v in right_pos.items()}
            right_pos = ensure_safe_goal_position(gp_right, self.config.max_relative_target)

        if left_pos:
            self.left_bus.sync_write("Goal_Position", {k.replace(".pos", ""): v for k, v in left_pos.items()})
        if right_pos:
            self.right_bus.sync_write("Goal_Position", {k.replace(".pos", ""): v for k, v in right_pos.items()})
        self.left_bus.sync_write("Goal_Velocity", base_wheel_goal_vel)

        return {**left_pos, **right_pos, **base_goal_vel}

    def stop_base(self):
        self.left_bus.sync_write("Goal_Velocity", dict.fromkeys(self.base_motors, 0), num_retry=0)
        logger.info("Base motors stopped")

    def read_and_check_currents(self, limit_ma, print_currents):
        scale = 6.5
        left_curr_raw = self.left_bus.sync_read("Present_Current", list(self.left_bus.motors.keys()))
        right_curr_raw = self.right_bus.sync_read("Present_Current", list(self.right_bus.motors.keys()))

        if print_currents:
            left_line = "{" + ",".join(str(int(v * scale)) for v in left_curr_raw.values()) + "}"
            print(f"Left Bus currents: {left_line}")
            right_line = "{" + ",".join(str(int(v * scale)) for v in right_curr_raw.values()) + "}"
            print(f"Right Bus currents: {right_line}")

        for name, raw in {**left_curr_raw, **right_curr_raw}.items():
            current_ma = float(raw) * scale
            if current_ma > limit_ma:
                print(f"[Overcurrent] {name}: {current_ma:.1f} mA > {limit_ma:.1f} mA, disconnecting!")
                try:
                    self.stop_base()
                except Exception:
                    pass
                try:
                    self.disconnect()
                except Exception as e:
                    print(f"[Overcurrent] disconnect error: {e}")
                sys.exit(1)

        return {k: round(v * scale, 1) for k, v in {**left_curr_raw, **right_curr_raw}.items()}

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.stop_base()
        self.left_bus.disconnect(self.config.disable_torque_on_disconnect)
        self.right_bus.disconnect(self.config.disable_torque_on_disconnect)
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
