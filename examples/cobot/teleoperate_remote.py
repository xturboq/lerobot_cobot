#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cobot 远程遥操作脚本 (局域网模式)
=================================
- PC端通过 ZMQ 连接机器人端的 CobotHost
- 双臂主臂 (BiSO100Leader) 控制双臂从臂
- 键盘控制四轮麦克纳姆轮底盘
- 摄像头图像通过网络传输并在 Rerun 中显示

架构说明:
    ┌─────────────────┐        ZMQ         ┌─────────────────┐
    │     PC 端       │  <=============>   │   Jetson 端     │
    │  (主臂+键盘)    │   局域网通信        │  (从臂+底盘)    │
    │  teleoperate_   │                    │  cobot_host     │
    │  remote.py      │                    │                 │
    └─────────────────┘                    └─────────────────┘

使用方式:
    1. 在机器人端 (Jetson) 运行:
       python -m lerobot.robots.cobot.cobot_host

    2. 在 PC 端运行:
       python examples/cobot/teleoperate_remote.py \
           --remote_ip=192.168.1.100 \
           --teleop_left_port=/dev/ttyACM1 \
           --teleop_right_port=/dev/ttyACM0 \
           --display_data

按键说明:
    W/S - 前进/后退
    A/D - 左移/右移
    Z/X - 左转/右转
    R/F - 加速/减速
    Q   - 退出
"""

import argparse
import time

from lerobot.robots.cobot import CobotClient, CobotClientConfig
from lerobot.teleoperators.bi_so100_leader import BiSO100Leader, BiSO100LeaderConfig
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cobot 远程遥操作 (局域网模式)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # 网络配置
    parser.add_argument("--remote_ip", type=str, required=True,
                        help="机器人 Host 的 IP 地址 (必填)")
    parser.add_argument("--port_cmd", type=int, default=5555,
                        help="ZMQ 命令端口 (默认: 5555)")
    parser.add_argument("--port_obs", type=int, default=5556,
                        help="ZMQ 观测端口 (默认: 5556)")
    
    # 主臂串口配置 (PC 端)
    parser.add_argument("--teleop_left_port", type=str, default="/dev/ttyACM1",
                        help="左主臂串口 (默认: /dev/ttyACM1)")
    parser.add_argument("--teleop_right_port", type=str, default="/dev/ttyACM0",
                        help="右主臂串口 (默认: /dev/ttyACM0)")
    
    # 其他配置
    parser.add_argument("--fps", type=int, default=30,
                        help="主循环频率 Hz (默认: 30)")
    parser.add_argument("--display_data", action="store_true",
                        help="在 Rerun 中显示数据")
    
    return parser.parse_args()


class RemoteTeleoperator:
    """远程遥操作控制器"""
    
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
    
    def __init__(self, robot: CobotClient, leader: BiSO100Leader, keyboard: KeyboardTeleop):
        self.robot = robot
        self.leader = leader
        self.keyboard = keyboard
        self.speed_index = 1  # 默认中速
        
    def _keyboard_to_base_action(self, pressed_keys: dict) -> dict:
        """将键盘输入转换为底盘速度指令"""
        keys = set(pressed_keys.keys())
        
        # 速度档位调节
        if self.TELEOP_KEYS["speed_up"] in keys:
            self.speed_index = min(self.speed_index + 1, len(self.SPEED_LEVELS) - 1)
            print(f"\n⬆️ 速度档位: {self.speed_index + 1}")
        if self.TELEOP_KEYS["speed_down"] in keys:
            self.speed_index = max(self.speed_index - 1, 0)
            print(f"\n⬇️ 速度档位: {self.speed_index + 1}")
        
        speed = self.SPEED_LEVELS[self.speed_index]
        xy_speed = speed["xy"]
        theta_speed = speed["theta"]
        
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
            if key.startswith("left_") or key.startswith("right_"):
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
    print("    Cobot 远程遥操作 (局域网模式)")
    print("=" * 60)
    print(f"\n远程机器人: {args.remote_ip}:{args.port_cmd}/{args.port_obs}")
    print(f"主臂串口: 左={args.teleop_left_port}, 右={args.teleop_right_port}")
    print(f"循环频率: {args.fps} Hz")
    print("\n【按键说明】")
    print("  W/S    - 前进/后退")
    print("  A/D    - 左移/右移")
    print("  Z/X    - 左转/右转")
    print("  R/F    - 加速/减速")
    print("  Q      - 退出")
    print("=" * 60)
    
    # ============ 创建配置 ============
    # 远程机器人 (CobotClient)
    robot_config = CobotClientConfig(
        id="cobot_remote",
        remote_ip=args.remote_ip,
        port_zmq_cmd=args.port_cmd,
        port_zmq_observations=args.port_obs,
    )
    robot = CobotClient(robot_config)
    
    # 双臂主臂 (BiSO100Leader) - 连接在 PC 端
    leader_config = BiSO100LeaderConfig(
        id="cobot_leader",
        left_arm_port=args.teleop_left_port,
        right_arm_port=args.teleop_right_port,
    )
    leader = BiSO100Leader(leader_config)
    
    # 键盘
    keyboard_config = KeyboardTeleopConfig(id="keyboard")
    keyboard = KeyboardTeleop(keyboard_config)
    
    # ============ 连接设备 ============
    print("\n正在连接设备...")
    
    try:
        print(f"正在连接远程机器人 {args.remote_ip}...")
        robot.connect()
        print("✅ 远程机器人 (CobotHost) 已连接")
    except Exception as e:
        print(f"❌ 远程机器人连接失败: {e}")
        print("   请确保机器人端已运行: python -m lerobot.robots.cobot.cobot_host")
        return
    
    try:
        leader.connect()
        print("✅ 主臂 已连接")
    except Exception as e:
        print(f"❌ 主臂 连接失败: {e}")
        robot.disconnect()
        return
    
    keyboard.connect()
    print("✅ 键盘 已连接")
    
    # 初始化 Rerun 可视化
    if args.display_data:
        init_rerun(session_name="cobot_teleop_remote")
        print("✅ Rerun 可视化已启用")
    
    # 创建控制器
    teleop = RemoteTeleoperator(robot, leader, keyboard)
    
    print("\n🚀 开始远程遥操作! 按 Q 退出...\n")
    
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
            
            # 7. 发送动作到远程机器人
            robot.send_action(action)
            
            # 8. 获取远程观测并显示
            obs = robot.get_observation()
            
            # 简洁的状态显示
            print(f"\r底盘: x={base_action['x.vel']:+.2f} y={base_action['y.vel']:+.2f} θ={base_action['theta.vel']:+.1f}  档位:{teleop.speed_index+1}", end="")
            
            # 9. Rerun 可视化
            if args.display_data:
                log_rerun_data(observation=obs, action=action, compress_images=True)
            
            # 10. 控制循环频率
            precise_sleep(max(1.0 / args.fps - (time.perf_counter() - t0), 0.0))
            
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断 (Ctrl+C)")
    
    finally:
        # ============ 断开连接 ============
        print("\n正在断开连接...")
        keyboard.disconnect()
        leader.disconnect()
        robot.disconnect()
        print("✅ 所有设备已断开")
        print("\n程序结束")


if __name__ == "__main__":
    main()
