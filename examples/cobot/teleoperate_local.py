#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cobot 纯本地遥操作脚本
=====================
- 双臂主臂 (BiSO100Leader) 控制双臂从臂
- 键盘控制四轮麦克纳姆轮底盘
- 可选 Rerun 可视化

运行方式:
    python examples/cobot/teleoperate_local.py
    python examples/cobot/teleoperate_local.py --display_data  # 启用 Rerun 可视化

按键说明:
    W/S - 前进/后退
    A/D - 左移/右移
    Z/X - 左转/右转
    R/F - 加速/减速
    Q   - 退出
"""

import argparse
import time

from lerobot.robots.cobot import Cobot, CobotConfig
from lerobot.teleoperators.bi_so100_leader import BiSO100Leader, BiSO100LeaderConfig
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data


def parse_args():
    parser = argparse.ArgumentParser(description="Cobot 纯本地遥操作")
    parser.add_argument("--fps", type=int, default=30, help="主循环频率 (Hz)")
    parser.add_argument("--use_dummy", action="store_true", help="仅打印动作，不连接机器人")
    
    # 从臂串口配置
    parser.add_argument("--left_port", type=str, default="/dev/cobot_follow_left", 
                        help="左总线串口 (左臂+底盘)")
    parser.add_argument("--right_port", type=str, default="/dev/cobot_follow_right", 
                        help="右总线串口 (右臂)")
    
    # 主臂串口配置
    parser.add_argument("--leader_left_port", type=str, default="/dev/cobot_leader_left",
                        help="左主臂串口")
    parser.add_argument("--leader_right_port", type=str, default="/dev/cobot_leader_right",
                        help="右主臂串口")
    
    # Rerun 可视化
    parser.add_argument("--display_data", action="store_true",
                        help="在 Rerun 中显示数据")
    
    # 摄像头配置 (用于可视化)
    parser.add_argument("--cam_front", type=str, default="/dev/video0",
                        help="前置摄像头设备")
    parser.add_argument("--cam_wrist", type=str, default="/dev/video1",
                        help="腕部摄像头设备")
    parser.add_argument("--no_cameras", action="store_true",
                        help="禁用摄像头")
    
    return parser.parse_args()


class LocalTeleoperator:
    """本地遥操作控制器"""
    
    # 键盘映射
    TELEOP_KEYS = {
        "forward": "w",
        "backward": "s",
        "left": "a",
        "right": "d",
        "rotate_left": "z",
        "rotate_right": "x",
        "speed_up": "r",
        "speed_down": "f",
        "quit": "q",
    }
    
    # 速度档位
    SPEED_LEVELS = [
        {"xy": 0.10, "theta": 30},   # 慢速
        {"xy": 0.15, "theta": 45},   # 中速
        {"xy": 0.20, "theta": 60},   # 快速
    ]
    
    def __init__(self, robot: Cobot, leader: BiSO100Leader, keyboard: KeyboardTeleop):
        self.robot = robot
        self.leader = leader
        self.keyboard = keyboard
        self.speed_index = 1  # 默认中速
        
    def _keyboard_to_base_action(self, pressed_keys: dict) -> dict:
        """
        将键盘输入转换为底盘速度指令
        
        Args:
            pressed_keys: KeyboardTeleop.get_action() 返回的字典，键是按下的键名
        """
        # 获取当前按下的键集合
        keys = set(pressed_keys.keys())
        
        # 速度档位调节
        if self.TELEOP_KEYS["speed_up"] in keys:
            self.speed_index = min(self.speed_index + 1, len(self.SPEED_LEVELS) - 1)
            print(f"\n⬆️ 速度档位: {self.speed_index + 1}")
        if self.TELEOP_KEYS["speed_down"] in keys:
            self.speed_index = max(self.speed_index - 1, 0)
            print(f"\n⬇️ 速度档位: {self.speed_index + 1}")
        
        speed = self.SPEED_LEVELS[self.speed_index]
        xy_speed = speed["xy"]      # m/s
        theta_speed = speed["theta"]  # deg/s
        
        x_vel = 0.0
        y_vel = 0.0
        theta_vel = 0.0
        
        if self.TELEOP_KEYS["forward"] in keys:
            x_vel += xy_speed
        if self.TELEOP_KEYS["backward"] in keys:
            x_vel -= xy_speed
        if self.TELEOP_KEYS["left"] in keys:
            y_vel += xy_speed
        if self.TELEOP_KEYS["right"] in keys:
            y_vel -= xy_speed
        if self.TELEOP_KEYS["rotate_left"] in keys:
            theta_vel += theta_speed
        if self.TELEOP_KEYS["rotate_right"] in keys:
            theta_vel -= theta_speed
            
        return {
            "x.vel": x_vel,
            "y.vel": y_vel,
            "theta.vel": theta_vel,
        }
    
    def _map_leader_to_follower(self, leader_action: dict) -> dict:
        """
        将主臂动作映射到从臂动作
        
        主臂格式: left_shoulder_pan.pos, right_shoulder_pan.pos
        从臂格式: arm_left_shoulder_pan.pos, arm_right_shoulder_pan.pos
        """
        follower_action = {}
        
        for key, value in leader_action.items():
            if key.startswith("left_"):
                # left_shoulder_pan.pos -> arm_left_shoulder_pan.pos
                new_key = "arm_" + key
            elif key.startswith("right_"):
                # right_shoulder_pan.pos -> arm_right_shoulder_pan.pos
                new_key = "arm_" + key
            else:
                new_key = key
            follower_action[new_key] = value
            
        return follower_action
    
    def should_quit(self, pressed_keys: dict) -> bool:
        """检查是否退出"""
        return self.TELEOP_KEYS["quit"] in pressed_keys.keys()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("    Cobot 纯本地遥操作")
    print("=" * 60)
    print("\n【按键说明】")
    print("  W/S    - 前进/后退")
    print("  A/D    - 左移/右移")
    print("  Z/X    - 左转/右转")
    print("  R/F    - 加速/减速")
    print("  Q      - 退出")
    print("=" * 60)
    
    # ============ 创建配置 ============
    # 摄像头配置 (仅在显示数据时启用)
    if args.display_data and not args.no_cameras:
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
        from lerobot.cameras.configs import Cv2Rotation
        cameras_config = {
            "cam_front": OpenCVCameraConfig(
                index_or_path=args.cam_front, fps=args.fps, width=640, height=480,
                rotation=Cv2Rotation.NO_ROTATION
            ),
            "cam_wrist": OpenCVCameraConfig(
                index_or_path=args.cam_wrist, fps=args.fps, width=640, height=480,
                rotation=Cv2Rotation.NO_ROTATION
            ),
        }
    else:
        cameras_config = {}
    
    # 从臂 + 底盘 (Cobot)
    robot_config = CobotConfig(
        id="cobot_follower",
        left_port=args.left_port,
        right_port=args.right_port,
        cameras=cameras_config,
    )
    robot = Cobot(robot_config)
    
    # 双臂主臂 (BiSO100Leader)
    leader_config = BiSO100LeaderConfig(
        id="cobot_leader",
        left_arm_port=args.leader_left_port,
        right_arm_port=args.leader_right_port,
    )
    leader = BiSO100Leader(leader_config)
    
    # 键盘
    keyboard_config = KeyboardTeleopConfig(id="keyboard")
    keyboard = KeyboardTeleop(keyboard_config)
    
    # ============ 连接设备 ============
    if not args.use_dummy:
        print("\n正在连接设备...")
        try:
            robot.connect()
            print("✅ 从臂+底盘 已连接")
        except Exception as e:
            print(f"❌ 从臂+底盘 连接失败: {e}")
            return
            
        try:
            leader.connect()
            print("✅ 主臂 已连接")
        except Exception as e:
            print(f"❌ 主臂 连接失败: {e}")
            robot.disconnect()
            return
    else:
        print("\n🧪 USE_DUMMY 模式: 仅打印动作")
    
    keyboard.connect()
    print("✅ 键盘 已连接")
    
    # 初始化 Rerun 可视化
    if args.display_data:
        init_rerun(session_name="cobot_teleop")
        print("✅ Rerun 可视化已启用")
    
    # 创建控制器
    teleop = LocalTeleoperator(robot, leader, keyboard)
    
    print("\n🚀 开始遥操作! 按 Q 退出...\n")
    
    # ============ 主循环 ============
    try:
        while True:
            t0 = time.perf_counter()
            
            # 1. 读取主臂位置
            leader_action = leader.get_action()
            
            # 2. 映射到从臂格式
            arm_action = teleop._map_leader_to_follower(leader_action)
            
            # 3. 读取键盘输入
            pressed_keys = keyboard.get_action()
            
            # 4. 检查退出
            if teleop.should_quit(pressed_keys):
                print("\n按下 Q，退出遥操作...")
                break
            
            # 5. 计算底盘速度
            base_action = teleop._keyboard_to_base_action(pressed_keys)
            
            # 6. 合并动作
            action = {**arm_action, **base_action}
            
            # 7. 发送动作
            if args.use_dummy:
                # 简化输出
                left_pos = [f"{v:.1f}" for k, v in arm_action.items() if "left" in k]
                right_pos = [f"{v:.1f}" for k, v in arm_action.items() if "right" in k]
                print(f"\r左臂:{left_pos} 右臂:{right_pos} 底盘:x={base_action['x.vel']:.2f} y={base_action['y.vel']:.2f} θ={base_action['theta.vel']:.1f}", end="")
            else:
                robot.send_action(action)
                # 简洁的状态显示
                print(f"\r底盘: x={base_action['x.vel']:+.2f} y={base_action['y.vel']:+.2f} θ={base_action['theta.vel']:+.1f}  档位:{teleop.speed_index+1}", end="")
            
            # 8. Rerun 可视化
            if args.display_data and not args.use_dummy:
                obs = robot.get_observation()
                log_rerun_data(observation=obs, action=action)
            
            # 9. 控制循环频率
            precise_sleep(max(1.0 / args.fps - (time.perf_counter() - t0), 0.0))
            
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断 (Ctrl+C)")
    
    finally:
        # ============ 断开连接 ============
        print("\n正在断开连接...")
        keyboard.disconnect()
        leader.disconnect()
        if not args.use_dummy:
            robot.disconnect()
        print("✅ 所有设备已断开")
        print("\n程序结束")


if __name__ == "__main__":
    main()

