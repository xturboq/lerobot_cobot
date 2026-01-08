#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cobot 本地遥操作录制脚本
========================
- 双臂主臂 (BiSO100Leader) 控制双臂从臂
- 键盘控制四轮麦克纳姆轮底盘
- 摄像头采集图像

运行方式:
    python examples/cobot/record_local.py --dataset <username>/<dataset_name>

按键说明 (录制控制):
    Enter      - 结束当前回合，保存并开始下一回合
    Backspace  - 丢弃当前回合，重新录制
    Escape     - 停止录制，保存所有数据

按键说明 (底盘控制):
    W/S - 前进/后退
    A/D - 左移/右移
    Z/X - 左转/右转
    R/F - 加速/减速
"""

import argparse
import logging
from datetime import datetime

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import combine_feature_dicts
from lerobot.processor import make_default_processors
from lerobot.robots.cobot import Cobot, CobotConfig
from lerobot.scripts.lerobot_record import record_loop
from lerobot.teleoperators.bi_so100_leader import BiSO100Leader, BiSO100LeaderConfig
from lerobot.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun


def parse_args():
    parser = argparse.ArgumentParser(description="Cobot 本地遥操作录制")
    
    # 数据集配置
    parser.add_argument("--dataset", type=str, required=True,
                        help="数据集 repo_id, 例如: liyitenga/cobot_record_001")
    parser.add_argument("--num_episodes", type=int, default=10, 
                        help="录制回合数 (默认: 10)")
    parser.add_argument("--fps", type=int, default=30, 
                        help="采集频率 Hz (默认: 30)")
    parser.add_argument("--episode_time", type=int, default=60, 
                        help="每回合时长 秒 (默认: 60)")
    parser.add_argument("--reset_time", type=int, default=10, 
                        help="重置间隔 秒 (默认: 10)")
    parser.add_argument("--task_description", type=str, default="操作任务",
                        help="任务描述")
    
    # 机器人配置
    parser.add_argument("--robot_id", type=str, default="cobot_local",
                        help="机器人 ID")
    parser.add_argument("--left_port", type=str, default="/dev/cobot_follow_left",
                        help="左总线串口 (左臂+底盘)")
    parser.add_argument("--right_port", type=str, default="/dev/cobot_follow_right",
                        help="右总线串口 (右臂)")
    
    # 主臂配置
    parser.add_argument("--leader_id", type=str, default="cobot_leader",
                        help="主臂 ID")
    parser.add_argument("--leader_left_port", type=str, default="/dev/cobot_leader_left",
                        help="左主臂串口")
    parser.add_argument("--leader_right_port", type=str, default="/dev/cobot_leader_right",
                        help="右主臂串口")
    
    # 摄像头配置
    parser.add_argument("--cam_front", type=str, default="/dev/video0",
                        help="前置摄像头设备")
    parser.add_argument("--cam_wrist", type=str, default="/dev/video1",
                        help="腕部摄像头设备")
    parser.add_argument("--no_cameras", action="store_true",
                        help="禁用摄像头")
    
    # 其他
    parser.add_argument("--display_data", action="store_true", default=True,
                        help="在 Rerun 中显示数据")
    parser.add_argument("--push_to_hub", action="store_true",
                        help="完成后推送到 HuggingFace Hub")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    logging.basicConfig(level=logging.INFO, format="[%(filename)s:%(lineno)d] %(message)s")
    
    print("=" * 60)
    print("    Cobot 本地遥操作录制")
    print("=" * 60)
    print(f"\n数据集: {args.dataset}")
    print(f"录制: {args.num_episodes} 回合, 每回合 {args.episode_time} 秒")
    print(f"采集频率: {args.fps} Hz")
    print("\n【录制控制按键】")
    print("  Enter      - 结束当前回合，保存并开始下一回合")
    print("  Backspace  - 丢弃当前回合，重新录制")
    print("  Escape     - 停止录制，保存所有数据")
    print("\n【底盘控制按键】")
    print("  W/S  - 前进/后退")
    print("  A/D  - 左移/右移")
    print("  Z/X  - 左转/右转")
    print("  R/F  - 加速/减速")
    print("=" * 60)
    
    # ============ 创建配置 ============
    # 摄像头配置
    if args.no_cameras:
        cameras_config = {}
    else:
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
    
    # 机器人配置 (从臂 + 底盘)
    robot_config = CobotConfig(
        id=args.robot_id,
        left_port=args.left_port,
        right_port=args.right_port,
        cameras=cameras_config,
    )
    robot = Cobot(robot_config)
    
    # 主臂配置 (双臂)
    leader_config = BiSO100LeaderConfig(
        id=args.leader_id,
        left_arm_port=args.leader_left_port,
        right_arm_port=args.leader_right_port,
    )
    leader_arm = BiSO100Leader(leader_config)
    
    # 键盘
    keyboard_config = KeyboardTeleopConfig(id="keyboard")
    keyboard = KeyboardTeleop(keyboard_config)
    
    # 处理器
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()
    
    # ============ 数据集配置 ============
    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=True,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=True,
        ),
    )
    
    dataset = LeRobotDataset.create(
        repo_id=args.dataset,
        fps=args.fps,
        features=dataset_features,
        robot_type=robot.name,
        use_videos=True,
        image_writer_processes=1,
        image_writer_threads=4 * len(cameras_config) if cameras_config else 1,
    )
    logging.info(f"数据集已创建: {dataset.repo_id}")
    
    # ============ 连接设备 ============
    print("\n正在连接设备...")
    try:
        robot.connect()
        logging.info("✅ 从臂+底盘 已连接")
    except Exception as e:
        logging.error(f"❌ 从臂+底盘 连接失败: {e}")
        return
    
    try:
        leader_arm.connect()
        logging.info("✅ 主臂 已连接")
    except Exception as e:
        logging.error(f"❌ 主臂 连接失败: {e}")
        robot.disconnect()
        return
    
    keyboard.connect()
    logging.info("✅ 键盘 已连接")
    
    # 初始化键盘监听器 (用于录制控制)
    listener, events = init_keyboard_listener()
    
    # 初始化 Rerun 可视化
    if args.display_data:
        init_rerun(session_name="cobot_record")
    
    # 检查连接
    if not robot.is_connected or not leader_arm.is_connected or not keyboard.is_connected:
        raise ValueError("设备连接失败!")
    
    print("\n🚀 开始录制!\n")
    
    # ============ 录制循环 ============
    recorded_episodes = 0
    
    try:
        while recorded_episodes < args.num_episodes and not events["stop_recording"]:
            log_say(f"录制回合 {recorded_episodes + 1} / {args.num_episodes}")
            
            # 主录制循环
            record_loop(
                robot=robot,
                events=events,
                fps=args.fps,
                dataset=dataset,
                teleop=[leader_arm, keyboard],
                control_time_s=args.episode_time,
                single_task=args.task_description,
                display_data=args.display_data,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
            )
            
            # 重置环境
            if not events["stop_recording"] and (
                (recorded_episodes < args.num_episodes - 1) or events["rerecord_episode"]
            ):
                log_say("重置环境")
                record_loop(
                    robot=robot,
                    events=events,
                    fps=args.fps,
                    teleop=[leader_arm, keyboard],
                    control_time_s=args.reset_time,
                    single_task=args.task_description,
                    display_data=args.display_data,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                )
            
            # 检查是否需要重录
            if events["rerecord_episode"]:
                log_say("重新录制本回合")
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                continue
            
            # 保存回合
            dataset.save_episode()
            recorded_episodes += 1
            logging.info(f"回合 {recorded_episodes} 已保存")
    
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断 (Ctrl+C)")
    
    finally:
        # ============ 清理 ============
        print("\n正在清理...")
        log_say("停止录制")
        
        robot.disconnect()
        leader_arm.disconnect()
        keyboard.disconnect()
        listener.stop()
        
        # 完成数据集
        dataset.finalize()
        logging.info(f"数据集已保存: {dataset.repo_id}")
        
        # 推送到 Hub
        if args.push_to_hub:
            logging.info("正在推送到 HuggingFace Hub...")
            dataset.push_to_hub()
            logging.info("推送完成!")
        
        print("\n✅ 所有设备已断开")
        print(f"录制完成: {recorded_episodes} 回合")
        print(f"数据集: {dataset.repo_id}")


if __name__ == "__main__":
    main()

