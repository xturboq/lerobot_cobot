#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time
from functools import cached_property
from itertools import chain
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
from .config_lekiwi import LeKiwiConfig

logger = logging.getLogger(__name__)


class LeKiwi(Robot):
    """
    双臂 + 四轮麦克纳姆轮移动底盘
    左总线: 左臂(7) + 底盘(4)
    右总线: 右臂(7)
    """

    config_class = LeKiwiConfig
    name = "lekiwi"

    def __init__(self, config: LeKiwiConfig):
        super().__init__(config)
        self.config = config
        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100

        # 左总线: 左臂 + 底盘
        self.left_bus = FeetechMotorsBus(
            port=self.config.left_port,
            motors={
                # 左机械臂
                "arm_left_shoulder_pan": Motor(1, "sts3215", norm_mode_body),
                "arm_left_shoulder_lift": Motor(2, "sts3215", norm_mode_body),
                "arm_left_shoulder_rotate": Motor(3, "sts3215", norm_mode_body),
                "arm_left_elbow_flex": Motor(4, "sts3215", norm_mode_body),
                "arm_left_wrist_flex": Motor(5, "sts3215", norm_mode_body),
                "arm_left_wrist_roll": Motor(6, "sts3215", norm_mode_body),
                "arm_left_gripper": Motor(7, "sts3215", MotorNormMode.RANGE_0_100),
                # 四轮麦克纳姆轮底盘
                "base_fl": Motor(21, "sts3215", MotorNormMode.RANGE_M100_100),  # 左前
                "base_fr": Motor(22, "sts3215", MotorNormMode.RANGE_M100_100),  # 右前
                "base_rl": Motor(23, "sts3215", MotorNormMode.RANGE_M100_100),  # 左后
                "base_rr": Motor(24, "sts3215", MotorNormMode.RANGE_M100_100),  # 右后
            },
            calibration=self.calibration,
        )

        # 右总线: 右臂
        self.right_bus = FeetechMotorsBus(
            port=self.config.right_port,
            motors={
                # 右机械臂
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

        # 底盘几何参数
        self.wheel_radius = config.wheel_radius
        self.lx = config.lx
        self.ly = config.ly

    @property
    def _state_ft(self) -> dict[str, type]:
        """定义状态特征: 双臂位置 + 底盘速度"""
        return dict.fromkeys(
            (
                # 左臂
                "arm_left_shoulder_pan.pos",
                "arm_left_shoulder_lift.pos",
                "arm_left_shoulder_rotate.pos",
                "arm_left_elbow_flex.pos",
                "arm_left_wrist_flex.pos",
                "arm_left_wrist_roll.pos",
                "arm_left_gripper.pos",
                # 右臂
                "arm_right_shoulder_pan.pos",
                "arm_right_shoulder_lift.pos",
                "arm_right_shoulder_rotate.pos",
                "arm_right_elbow_flex.pos",
                "arm_right_wrist_flex.pos",
                "arm_right_wrist_roll.pos",
                "arm_right_gripper.pos",
                # 底盘
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
        双臂 + 底盘校准:
        - 左臂: 位置模式 → 半圈归零 → 记录ROM
        - 底盘: 无需归零; ROM固定为 0–4095
        - 右臂: 位置模式 → 半圈归零 → 记录ROM
        """
        # 如果已有校准文件: 加载并写回
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

        # === 左臂校准 ===
        self.left_bus.disable_torque(self.left_arm_motors)
        for name in self.left_arm_motors:
            self.left_bus.write("Operating_Mode", name, OperatingMode.POSITION.value)

        input("Move LEFT arm to the middle of its range of motion, then press ENTER...")
        left_homing = self.left_bus.set_half_turn_homings(self.left_arm_motors)

        # 底盘轮子固定为全圈
        for wheel in self.base_motors:
            left_homing[wheel] = 0

        motors_left_all = self.left_arm_motors + self.base_motors
        # 底盘轮子和 wrist_roll 是全圈电机
        full_turn_left = [m for m in motors_left_all if m.startswith("base_") or "wrist_roll" in m]
        unknown_left = [m for m in motors_left_all if m not in full_turn_left]

        print("Move LEFT arm joints sequentially through full ROM. Press ENTER to stop...")
        l_mins, l_maxs = self.left_bus.record_ranges_of_motion(unknown_left)
        for m in full_turn_left:
            l_mins[m] = 0
            l_maxs[m] = 4095

        # === 右臂校准 ===
        self.right_bus.disable_torque(self.right_arm_motors)
        for name in self.right_arm_motors:
            self.right_bus.write("Operating_Mode", name, OperatingMode.POSITION.value)

        input("Move RIGHT arm to the middle of its range of motion, then press ENTER...")
        right_homing = self.right_bus.set_half_turn_homings(self.right_arm_motors)

        # wrist_roll 是全圈电机，不需要记录 ROM
        full_turn_right = [m for m in self.right_arm_motors if "wrist_roll" in m]
        unknown_right = [m for m in self.right_arm_motors if m not in full_turn_right]

        print("Move RIGHT arm joints sequentially through full ROM. Press ENTER to stop...")
        r_mins, r_maxs = self.right_bus.record_ranges_of_motion(unknown_right)
        for m in full_turn_right:
            r_mins[m] = 0
            r_maxs[m] = 4095

        # === 合并校准数据 ===
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

        # 写回校准
        calib_left = {k: v for k, v in self.calibration.items() if k in self.left_bus.motors}
        self.left_bus.write_calibration(calib_left, cache=False)
        self.left_bus.calibration = calib_left

        calib_right = {k: v for k, v in self.calibration.items() if k in self.right_bus.motors}
        self.right_bus.write_calibration(calib_right, cache=False)
        self.right_bus.calibration = calib_right

        self._save_calibration()
        print("Calibration saved to", self.calibration_fpath)

    def configure(self):
        """配置电机参数"""
        # 左臂: 位置模式
        self.left_bus.disable_torque()
        self.left_bus.configure_motors()
        for name in self.left_arm_motors:
            self.left_bus.write("Operating_Mode", name, OperatingMode.POSITION.value)
            self.left_bus.write("P_Coefficient", name, 16)
            self.left_bus.write("I_Coefficient", name, 0)
            self.left_bus.write("D_Coefficient", name, 32)

        # 底盘: 速度模式
        for name in self.base_motors:
            self.left_bus.write("Operating_Mode", name, OperatingMode.VELOCITY.value)

        self.left_bus.enable_torque()

        # 右臂: 位置模式
        self.right_bus.disable_torque()
        self.right_bus.configure_motors()
        for name in self.right_arm_motors:
            self.right_bus.write("Operating_Mode", name, OperatingMode.POSITION.value)
            self.right_bus.write("P_Coefficient", name, 16)
            self.right_bus.write("I_Coefficient", name, 0)
            self.right_bus.write("D_Coefficient", name, 32)

        self.right_bus.enable_torque()

    def setup_motors(self) -> None:
        # 左总线电机设置
        print("=== Setting up LEFT bus motors (left arm + base) ===")
        for motor in chain(reversed(self.left_arm_motors), reversed(self.base_motors)):
            input(f"Connect the LEFT controller board to the '{motor}' motor only and press enter.")
            self.left_bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.left_bus.motors[motor].id}")

        # 右总线电机设置
        print("=== Setting up RIGHT bus motors (right arm) ===")
        for motor in reversed(self.right_arm_motors):
            input(f"Connect the RIGHT controller board to the '{motor}' motor only and press enter.")
            self.right_bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.right_bus.motors[motor].id}")

    @staticmethod
    def _degps_to_raw(degps: float) -> int:
        steps_per_deg = 4096.0 / 360.0
        speed_in_steps = degps * steps_per_deg
        speed_int = int(round(speed_in_steps))
        # Cap the value to fit within signed 16-bit range (-32768 to 32767)
        if speed_int > 0x7FFF:
            speed_int = 0x7FFF  # 32767 -> maximum positive value
        elif speed_int < -0x8000:
            speed_int = -0x8000  # -32768 -> minimum negative value
        return speed_int

    @staticmethod
    def _raw_to_degps(raw_speed: int) -> float:
        steps_per_deg = 4096.0 / 360.0
        magnitude = raw_speed
        degps = magnitude / steps_per_deg
        return degps

    def _inverse_kinematics(self, vx: float, vy: float, omega: float) -> tuple[float, float, float, float]:
        """
        逆运动学：底盘速度 -> 轮子线速度
        
        Args:
            vx: 前进速度 (m/s)，正值前进
            vy: 侧向速度 (m/s)，正值左移
            omega: 旋转角速度 (rad/s)，正值逆时针
            
        Returns:
            (v_fl, v_fr, v_rl, v_rr): 四个轮子的线速度 (m/s)
        """
        K = self.lx + self.ly
        
        v_fl = vx - vy - K * omega  # 左前
        v_fr = vx + vy + K * omega  # 右前
        v_rl = vx + vy - K * omega  # 左后
        v_rr = vx - vy + K * omega  # 右后
        
        return v_fl, v_fr, v_rl, v_rr

    def _forward_kinematics(
        self, v_fl: float, v_fr: float, v_rl: float, v_rr: float
    ) -> tuple[float, float, float]:
        """
        正运动学：轮子线速度 -> 底盘速度
        
        Args:
            v_fl, v_fr, v_rl, v_rr: 四个轮子的线速度 (m/s)
            
        Returns:
            (vx, vy, omega): 底盘速度
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
        将机体坐标系速度转换为轮子原始指令
        
        Parameters:
            x: 前进速度 (m/s)
            y: 左右速度 (m/s)
            theta: 旋转速度 (deg/s)
            max_raw: 最大原始指令值
        
        Returns:
            {"base_fl": value, "base_fr": value, "base_rl": value, "base_rr": value}
        """
        # 转换旋转速度从 deg/s 到 rad/s
        theta_rad = theta * (np.pi / 180.0)
        
        # 逆运动学: 底盘速度 -> 轮子线速度 (m/s)
        v_fl, v_fr, v_rl, v_rr = self._inverse_kinematics(x, y, theta_rad)
        
        # 应用电机方向修正 (左右两侧电机安装方向相反)
        dirs = self.config.motor_directions
        v_fl *= dirs["base_fl"]
        v_fr *= dirs["base_fr"]
        v_rl *= dirs["base_rl"]
        v_rr *= dirs["base_rr"]
        
        # 线速度 -> 角速度 (rad/s)
        w_fl = v_fl / self.wheel_radius
        w_fr = v_fr / self.wheel_radius
        w_rl = v_rl / self.wheel_radius
        w_rr = v_rr / self.wheel_radius
        
        # 转换为 deg/s
        wheel_degps = np.array([w_fl, w_fr, w_rl, w_rr]) * (180.0 / np.pi)

        # 缩放以避免超过最大值
        steps_per_deg = 4096.0 / 360.0
        raw_floats = [abs(degps) * steps_per_deg for degps in wheel_degps]
        max_raw_computed = max(raw_floats)
        if max_raw_computed > max_raw:
            scale = max_raw / max_raw_computed
            wheel_degps = wheel_degps * scale

        # 转换为原始整数值
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
        将轮子原始反馈转换为机体坐标系速度
        
        Returns:
            {"x.vel": float, "y.vel": float, "theta.vel": float}
        """
        # 转换原始值为 deg/s
        wheel_degps = np.array(
            [
                self._raw_to_degps(fl_speed),
                self._raw_to_degps(fr_speed),
                self._raw_to_degps(rl_speed),
                self._raw_to_degps(rr_speed),
            ]
        )

        # 转换为 rad/s
        wheel_radps = wheel_degps * (np.pi / 180.0)
        # 角速度 -> 线速度 (m/s)
        wheel_linear_speeds = wheel_radps * self.wheel_radius

        # 应用电机方向修正 (读取反馈时也需要修正)
        dirs = self.config.motor_directions
        wheel_linear_speeds[0] *= dirs["base_fl"]
        wheel_linear_speeds[1] *= dirs["base_fr"]
        wheel_linear_speeds[2] *= dirs["base_rl"]
        wheel_linear_speeds[3] *= dirs["base_rr"]

        # 正运动学: 轮子线速度 -> 底盘速度
        vx, vy, omega_rad = self._forward_kinematics(
            wheel_linear_speeds[0],
            wheel_linear_speeds[1],
            wheel_linear_speeds[2],
            wheel_linear_speeds[3],
        )

        # 转换 omega 从 rad/s 到 deg/s
        theta = omega_rad * (180.0 / np.pi)
        return {
            "x.vel": vx,
            "y.vel": vy,
            "theta.vel": theta,
        }

    def get_observation(self) -> dict[str, Any]:
        """获取观测数据: 双臂位置 + 底盘速度 + 摄像头图像"""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start = time.perf_counter()

        # 读取左臂位置
        left_pos = self.left_bus.sync_read("Present_Position", self.left_arm_motors)
        # 读取底盘速度
        base_wheel_vel = self.left_bus.sync_read("Present_Velocity", self.base_motors)
        base_vel = self._wheel_raw_to_body(
            base_wheel_vel["base_fl"],
            base_wheel_vel["base_fr"],
            base_wheel_vel["base_rl"],
            base_wheel_vel["base_rr"],
        )
        # 读取右臂位置
        right_pos = self.right_bus.sync_read("Present_Position", self.right_arm_motors)

        # 组装观测字典
        left_arm_state = {f"{k}.pos": v for k, v in left_pos.items()}
        right_arm_state = {f"{k}.pos": v for k, v in right_pos.items()}
        obs_dict = {**left_arm_state, **right_arm_state, **base_vel}

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # 读取摄像头图像
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """发送动作指令: 双臂位置 + 底盘速度"""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # 分离左右臂位置和底盘速度
        left_pos = {k: v for k, v in action.items() if k.endswith(".pos") and k.startswith("arm_left_")}
        right_pos = {k: v for k, v in action.items() if k.endswith(".pos") and k.startswith("arm_right_")}
        base_goal_vel = {k: v for k, v in action.items() if k.endswith(".vel")}

        # 转换底盘速度为轮子原始指令
        base_wheel_goal_vel = self._body_to_wheel_raw(
            base_goal_vel.get("x.vel", 0.0),
            base_goal_vel.get("y.vel", 0.0),
            base_goal_vel.get("theta.vel", 0.0),
        )

        # 安全检查
        if left_pos and self.config.max_relative_target is not None:
            present_left = self.left_bus.sync_read("Present_Position", self.left_arm_motors)
            gp_left = {k: (v, present_left[k.replace(".pos", "")]) for k, v in left_pos.items()}
            left_pos = ensure_safe_goal_position(gp_left, self.config.max_relative_target)

        if right_pos and self.config.max_relative_target is not None:
            present_right = self.right_bus.sync_read("Present_Position", self.right_arm_motors)
            gp_right = {k: (v, present_right[k.replace(".pos", "")]) for k, v in right_pos.items()}
            right_pos = ensure_safe_goal_position(gp_right, self.config.max_relative_target)

        # 发送指令
        if left_pos:
            self.left_bus.sync_write("Goal_Position", {k.replace(".pos", ""): v for k, v in left_pos.items()})
        if right_pos:
            self.right_bus.sync_write("Goal_Position", {k.replace(".pos", ""): v for k, v in right_pos.items()})
        self.left_bus.sync_write("Goal_Velocity", base_wheel_goal_vel)

        return {**left_pos, **right_pos, **base_goal_vel}

    def stop_base(self):
        """停止底盘"""
        self.left_bus.sync_write("Goal_Velocity", dict.fromkeys(self.base_motors, 0), num_retry=5)
        logger.info("Base motors stopped")

    def disconnect(self):
        """断开连接"""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.stop_base()
        
        # 断开时可能会有 Overload error，加错误处理避免崩溃
        try:
            self.left_bus.disconnect(self.config.disable_torque_on_disconnect)
        except Exception as e:
            logger.warning(f"Left bus disconnect error (ignored): {e}")
        
        try:
            self.right_bus.disconnect(self.config.disable_torque_on_disconnect)
        except Exception as e:
            logger.warning(f"Right bus disconnect error (ignored): {e}")
        
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
