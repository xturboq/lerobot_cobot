#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cobot 远程双臂遥操作录制脚本 (局域网模式)
==========================================
- PC端通过 ZMQ 连接机器人端的 CobotHost
- 双臂主臂 (BiSO100Leader) 控制双臂从臂
- 键盘控制四轮麦克纳姆轮底盘
- 摄像头图像通过网络传输

使用方式:
    1. 在机器人端 (Jetson) 运行:
       # 双摄像头 (默认)
       python -m lerobot.robots.cobot.cobot_host
       
       # 单摄像头
       python -m lerobot.robots.cobot.cobot_host --cameras cam_front:/dev/video0

    2. 在 PC 端运行 (参数风格类似 lerobot-record):
       # 双摄像头 (默认)
       python examples/cobot/record_bi_remote.py \
           --robot.remote_ip=192.168.1.100 \
           --robot.id=cobot_remote \
           --teleop.left_port=/dev/ttyUSB0 \
           --teleop.right_port=/dev/ttyUSB1 \
           --teleop.id=cobot_leader \
           --display_data=true \
           --dataset.repo_id=coola/test \
           --dataset.num_episodes=5 \
           --dataset.single_task="Grab the black cube" \
           --dataset.push_to_hub=false \
           --dataset.episode_time_s=30 \
           --dataset.reset_time_s=20 \
           --dataset.fps=30
       
       # 单摄像头 (需与 Host 端配置一致)
       python examples/cobot/record_bi_remote.py \
           --robot.remote_ip=192.168.1.100 \
           --robot.cameras cam_front:/dev/video0 \
           ...

按键说明 (录制控制):
    → (右方向键) - 结束当前回合，保存并开始下一回合
    ← (左方向键) - 丢弃当前回合，重新录制
    Escape       - 停止录制，保存所有数据

按键说明 (底盘控制):
    W/S - 前进/后退
    A/D - 左移/右移
    Z/X - 左转/右转
    R/F - 加速/减速
"""

from lerobot.cameras.configs import Cv2Rotation
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.processor import make_default_processors
from lerobot.robots.cobot import CobotClient, CobotClientConfig
from lerobot.scripts.lerobot_record import record_loop
from lerobot.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.teleoperators.bi_so100_leader import BiSO100Leader, BiSO100LeaderConfig
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun

import argparse


def str_to_bool(value):
    """将字符串转换为布尔值"""
    if isinstance(value, bool):
        return value
    if value.lower() in ('true', '1', 'yes'):
        return True
    elif value.lower() in ('false', '0', 'no'):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected, got '{value}'")


def main():
    parser = argparse.ArgumentParser(
        description="Cobot 远程双臂遥操作录制 (局域网模式)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # ============ Robot 配置 (类似 --robot.xxx) ============
    robot_group = parser.add_argument_group('robot', '机器人配置')
    robot_group.add_argument("--robot.remote_ip", type=str, required=True,
                             dest="robot_remote_ip",
                             help="机器人 Host 的 IP 地址")
    robot_group.add_argument("--robot.id", type=str, default="cobot_client",
                             dest="robot_id",
                             help="机器人 ID")
    robot_group.add_argument("--robot.port_cmd", type=int, default=5555,
                             dest="robot_port_cmd",
                             help="ZMQ 命令端口 (默认: 5555)")
    robot_group.add_argument("--robot.port_obs", type=int, default=5556,
                             dest="robot_port_obs",
                             help="ZMQ 观测端口 (默认: 5556)")
    robot_group.add_argument("--robot.cameras", nargs="*", metavar="NAME:PATH",
                             dest="robot_cameras",
                             help="摄像头配置，格式: name:path (如 cam_front:/dev/video0)，需与 Host 端配置一致")
    
    # ============ Teleop 配置 (类似 --teleop.xxx) ============
    teleop_group = parser.add_argument_group('teleop', '主臂配置 (连接在PC端)')
    teleop_group.add_argument("--teleop.left_port", type=str, default="/dev/ttyUSB0",
                              dest="teleop_left_port",
                              help="左主臂串口")
    teleop_group.add_argument("--teleop.right_port", type=str, default="/dev/ttyUSB1",
                              dest="teleop_right_port",
                              help="右主臂串口")
    teleop_group.add_argument("--teleop.id", type=str, default="cobot_leader",
                              dest="teleop_id",
                              help="主臂 ID")
    
    # ============ Dataset 配置 (类似 --dataset.xxx) ============
    dataset_group = parser.add_argument_group('dataset', '数据集配置')
    dataset_group.add_argument("--dataset.repo_id", type=str, required=True,
                               dest="dataset_repo_id",
                               help="数据集 repo_id, 例如: coola/test")
    dataset_group.add_argument("--dataset.num_episodes", type=int, default=1,
                               dest="dataset_num_episodes",
                               help="录制回合数 (默认: 1)")
    dataset_group.add_argument("--dataset.fps", type=int, default=30,
                               dest="dataset_fps",
                               help="采集频率 Hz (默认: 30)")
    dataset_group.add_argument("--dataset.episode_time_s", type=int, default=60,
                               dest="dataset_episode_time_s",
                               help="每回合时长 秒 (默认: 60)")
    dataset_group.add_argument("--dataset.reset_time_s", type=int, default=10,
                               dest="dataset_reset_time_s",
                               help="重置间隔 秒 (默认: 10)")
    dataset_group.add_argument("--dataset.single_task", type=str, default="操作任务",
                               dest="dataset_single_task",
                               help="任务描述")
    dataset_group.add_argument("--dataset.push_to_hub", type=str_to_bool, default=False,
                               dest="dataset_push_to_hub",
                               help="完成后推送到 HuggingFace Hub (true/false)")
    
    # ============ 其他配置 ============
    parser.add_argument("--display_data", type=str_to_bool, default=True,
                        help="是否在 Rerun 中显示数据 (true/false)")
    parser.add_argument("--play_sounds", type=str_to_bool, default=True,
                        help="是否播放语音提示 (true/false)")
    
    args = parser.parse_args()

    print("=" * 60)
    print("    Cobot 远程双臂遥操作录制 (局域网模式)")
    print("=" * 60)
    print(f"\n数据集: {args.dataset_repo_id}")
    print(f"远程机器人: {args.robot_remote_ip}:{args.robot_port_cmd}/{args.robot_port_obs}")
    print(f"录制: {args.dataset_num_episodes} 回合, 每回合 {args.dataset_episode_time_s} 秒")
    print(f"采集频率: {args.dataset_fps} Hz")
    print(f"任务描述: {args.dataset_single_task}")
    print("\n【录制控制按键】")
    print("  → (右方向键) - 结束当前回合，保存并开始下一回合")
    print("  ← (左方向键) - 丢弃当前回合，重新录制")
    print("  Escape       - 停止录制，保存所有数据")
    print("\n【底盘控制按键】")
    print("  W/S  - 前进/后退")
    print("  A/D  - 左移/右移")
    print("  Z/X  - 左转/右转")
    print("  R/F  - 加速/减速")
    print("=" * 60)

    # === Robot and teleop config ===
    robot_config = CobotClientConfig(
        id=args.robot_id,
        remote_ip=args.robot_remote_ip,
        port_zmq_cmd=args.robot_port_cmd,
        port_zmq_observations=args.robot_port_obs,
    )
    
    # 处理摄像头配置 (需与 Host 端一致)
    if args.robot_cameras is not None:
        custom_cameras = {}
        for cam_spec in args.robot_cameras:
            if ":" not in cam_spec:
                print(f"错误: 摄像头配置格式应为 name:path，收到: {cam_spec}")
                return
            name, path = cam_spec.split(":", 1)
            custom_cameras[name] = OpenCVCameraConfig(
                index_or_path=path,
                fps=args.dataset_fps,
                width=640,
                height=480,
                rotation=Cv2Rotation.NO_ROTATION
            )
        robot_config.cameras = custom_cameras
        print(f"使用自定义摄像头配置: {list(custom_cameras.keys())}")
    
    leader_arm_config = BiSO100LeaderConfig(
        left_arm_port=args.teleop_left_port,
        right_arm_port=args.teleop_right_port,
        id=args.teleop_id,
    )
    keyboard_config = KeyboardTeleopConfig()

    robot = CobotClient(robot_config)
    leader_arm = BiSO100Leader(leader_arm_config)
    keyboard = KeyboardTeleop(keyboard_config)

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    # === Dataset setup ===
    action_features = hw_to_dataset_features(robot.action_features, ACTION)
    obs_features = hw_to_dataset_features(robot.observation_features, OBS_STR)
    dataset_features = {**action_features, **obs_features}

    dataset = LeRobotDataset.create(
        repo_id=args.dataset_repo_id,
        fps=args.dataset_fps,
        features=dataset_features,
        robot_type=robot.name,
        use_videos=True,
        image_writer_threads=4,
    )
    print(f"数据集已创建: {dataset.repo_id}")

    # === Connect devices ===
    print("\n正在连接设备...")
    
    try:
        print(f"正在连接远程机器人 {args.robot_remote_ip}...")
        robot.connect()
        print("✅ 远程机器人 (CobotHost) 已连接")
    except Exception as e:
        print(f"❌ 远程机器人连接失败: {e}")
        print("   请确保机器人端已运行: python -m lerobot.robots.cobot.cobot_host")
        return
    
    try:
        leader_arm.connect()
        print("✅ 主臂 已连接")
    except Exception as e:
        print(f"❌ 主臂 连接失败: {e}")
        robot.disconnect()
        return
    
    keyboard.connect()
    print("✅ 键盘 已连接")

    listener, events = init_keyboard_listener()
    
    if args.display_data:
        init_rerun(session_name="cobot_record_remote")

    if not robot.is_connected or not leader_arm.is_connected or not keyboard.is_connected:
        raise ValueError("设备连接失败!")

    print("\n🚀 开始录制!\n")
    recorded_episodes = 0

    try:
        while recorded_episodes < args.dataset_num_episodes and not events["stop_recording"]:
            log_say(f"录制回合 {recorded_episodes + 1} / {args.dataset_num_episodes}", args.play_sounds)

            # === Main record loop ===
            record_loop(
                robot=robot,
                events=events,
                fps=args.dataset_fps,
                dataset=dataset,
                teleop=[leader_arm, keyboard],
                control_time_s=args.dataset_episode_time_s,
                single_task=args.dataset_single_task,
                display_data=args.display_data,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
            )

            # === Reset environment ===
            if not events["stop_recording"] and (
                (recorded_episodes < args.dataset_num_episodes - 1) or events["rerecord_episode"]
            ):
                log_say("重置环境", args.play_sounds)
                record_loop(
                    robot=robot,
                    events=events,
                    fps=args.dataset_fps,
                    teleop=[leader_arm, keyboard],
                    control_time_s=args.dataset_reset_time_s,
                    single_task=args.dataset_single_task,
                    display_data=args.display_data,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                )

            if events["rerecord_episode"]:
                log_say("重新录制本回合", args.play_sounds)
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                continue

            dataset.save_episode()
            recorded_episodes += 1
            print(f"✅ 回合 {recorded_episodes} 已保存")

    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断 (Ctrl+C)")

    finally:
        # === Clean up ===
        print("\n正在清理...")
        log_say("停止录制", args.play_sounds)
        robot.disconnect()
        leader_arm.disconnect()
        keyboard.disconnect()
        listener.stop()
        dataset.finalize()
        
        if args.dataset_push_to_hub:
            print("正在推送到 HuggingFace Hub...")
            dataset.push_to_hub()
            print("✅ 推送完成!")
        
        print(f"\n✅ 录制完成: {recorded_episodes} 回合")
        print(f"数据集: {dataset.repo_id}")


if __name__ == "__main__":
    main()
