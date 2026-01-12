import dataclasses
from dataclasses import dataclass, field

from lerobot.cameras.configs import CameraConfig, Cv2Rotation
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

from ..config import RobotConfig


def dddbot_cameras_config() -> dict[str, CameraConfig]:
    return {
        "cam_front": OpenCVCameraConfig(
            index_or_path="/dev/video0", fps=30, width=640, height=480, rotation=Cv2Rotation.NO_ROTATION, fourcc="MJPG"
        ),
        "cam_wrist": OpenCVCameraConfig(
            index_or_path="/dev/video2", fps=30, width=640, height=480, rotation=Cv2Rotation.ROTATE_180, fourcc="MJPG"
        ),
    }


@RobotConfig.register_subclass("dddbot")
@dataclass
class DddBotConfig(RobotConfig):
    """DddBot Host Configuration - Dual Arm + 4-Wheel Mecanum Base"""
    left_port: str = "/dev/cobot_follow_left"  # Left Bus: Left Arm + Base
    right_port: str = "/dev/cobot_follow_right"  # Right Bus: Right Arm
    disable_torque_on_disconnect: bool = True

    # Base Geometry Parameters
    wheel_radius: float = 0.05  # Wheel radius (m)
    lx: float = 0.195  # Half of longitudinal wheelbase (m)
    ly: float = 0.248  # Half of lateral wheelbase (m)

    # Motor Directions (1=normal, -1=reversed)
    # Corrects for physical mounting of motors on opposite sides
    motor_directions: dict[str, int] = field(
        default_factory=lambda: {
            "base_fl": -1,  # Front Left
            "base_fr": 1,   # Front Right
            "base_rl": -1,  # Rear Left
            "base_rr": 1,   # Rear Right
        }
    )

    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    max_relative_target: int | None = None

    cameras: dict[str, CameraConfig] = field(default_factory=dddbot_cameras_config)

    # Set to `True` for backward compatibility with previous policies/dataset
    use_degrees: bool = False


@dataclass
class LeKiwiHostConfig:
    """LeKiwi Host Configuration (for DddBot)"""
    # Network Configuration
    port_zmq_cmd: int = 5555
    port_zmq_observations: int = 5556

    # Duration of the application
    connection_time_s: int = 6000

    # Watchdog: stop the robot if no command is received for over 1.5 seconds.
    watchdog_timeout_ms: int = 1500

    # If robot jitters decrease the frequency and monitor cpu load with `top` in cmd
    max_loop_freq_hz: int = 30


@RobotConfig.register_subclass("dddbot_client")
@dataclass
class DddBotClientConfig(RobotConfig):
    """DddBot Client Configuration - Remote Control"""
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

    cameras: dict[str, CameraConfig] = field(default_factory=dddbot_cameras_config)

    polling_timeout_ms: int = 15
    connect_timeout_s: int = 5
