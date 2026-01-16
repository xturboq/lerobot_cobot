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

from dataclasses import dataclass, field

from lerobot.cameras.configs import CameraConfig, Cv2Rotation
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

from ..config import RobotConfig


def lekiwi_cameras_config() -> dict[str, CameraConfig]:
    return {
        "front": OpenCVCameraConfig(
            index_or_path="/dev/video2", fps=30, width=640, height=480,fourcc="MJPG"
        ),
        "wrist": OpenCVCameraConfig(
            index_or_path="/dev/video0", fps=30, width=640, height=480, rotation=Cv2Rotation.ROTATE_180,fourcc="MJPG"
        ),
    }


@RobotConfig.register_subclass("lekiwi")
@dataclass
class LeKiwiConfig(RobotConfig):
    """Cobot Configuration - 双臂 + 四轮麦克纳姆轮底盘"""
    left_port: str = "/dev/cobot_follow_left"  # 左总线：左臂
    right_port: str = "/dev/cobot_follow_right"  # 右总线：右臂
    chassis_port: str = "/dev/cobot_chassis"  # 底盘总线：四轮麦克纳姆轮

    disable_torque_on_disconnect: bool = True

    # 底盘几何参数
    wheel_radius: float = 0.05  # 轮子半径 (m)
    lx: float = 0.195  # 前后轴距的一半 (m)
    ly: float = 0.248  # 左右轮距的一半 (m)

    # 底盘电机方向配置 (1=正向, -1=反向)
    motor_directions: dict[str, int] = field(
        default_factory=lambda: {
            "base_fl": -1,  # 左前轮
            "base_fr": 1,   # 右前轮
            "base_rl": -1,  # 左后轮
            "base_rr": 1,   # 右后轮
        }
    )

    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a dictionary that maps motor
    # names to the max_relative_target value for that motor.
    max_relative_target: float | dict[str, float] | None = None

    cameras: dict[str, CameraConfig] = field(default_factory=lekiwi_cameras_config)

    # Set to `True` for backward compatibility with previous policies/dataset
    use_degrees: bool = False


@dataclass
class LeKiwiHostConfig:
    # Network Configuration
    port_zmq_cmd: int = 5555
    port_zmq_observations: int = 5556

    # Duration of the application
    connection_time_s: int = 6000  # 约 100 分钟

    # Watchdog: stop the robot if no command is received for over 1.5 seconds.
    watchdog_timeout_ms: int = 1500

    # If robot jitters decrease the frequency and monitor cpu load with `top` in cmd
    max_loop_freq_hz: int = 30


@RobotConfig.register_subclass("lekiwi_client")
@dataclass
class LeKiwiClientConfig(RobotConfig):
    # Network Configuration
    remote_ip: str
    port_zmq_cmd: int = 5555
    port_zmq_observations: int = 5556

    teleop_keys: dict[str, str] = field(
        default_factory=lambda: {
            # Movement
            "forward": "w",
            "backward": "s",
            "left": "a",
            "right": "d",
            "rotate_left": "z",
            "rotate_right": "x",
            # Speed control
            "speed_up": "r",
            "speed_down": "f",
            # quit teleop
            "quit": "q",
        }
    )

    cameras: dict[str, CameraConfig] = field(default_factory=lekiwi_cameras_config)

    polling_timeout_ms: int = 15
    connect_timeout_s: int = 5
