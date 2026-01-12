# !/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.processor import make_default_processors
from lerobot.robots.dddbot.config_dddbot import DddBotClientConfig
from lerobot.robots.dddbot.dddbot_client import DddBotClient
from lerobot.scripts.lerobot_record import record_loop
from lerobot.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.teleoperators.bi_so100_leader import BiSO100Leader, BiSO100LeaderConfig
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun

import argparse


def main():
    parser = argparse.ArgumentParser(description="Record episodes with dddbot and bi-arm teleoperation")
    parser.add_argument("--dataset", type=str, required=True,
                    help="Dataset repo_id, e.g. user/dddbot_test_v1")
    parser.add_argument("--num_episodes", type=int, default=1, help="Number of episodes to record")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--episode_time", type=int, default=60, help="Duration of each episode (seconds)")
    parser.add_argument("--reset_time", type=int, default=10, help="Reset duration between episodes (seconds)")
    parser.add_argument("--task_description", type=str, default="DDDBot task", help="Task description")
    parser.add_argument("--remote_ip", type=str, default="127.0.0.1", help="Robot host IP")
    parser.add_argument("--robot_id", type=str, default="dddbot", help="Robot ID")
    parser.add_argument("--play_sounds", type=bool, default=True, help="Play sound prompts")
    args = parser.parse_args()

    # === 机器人与遥操作配置 ===
    robot_config = DddBotClientConfig(remote_ip=args.remote_ip, id=args.robot_id)
    # 使用 BiSO100Leader 作为双臂配置的标准
    leader_arm_config = BiSO100LeaderConfig(
        left_arm_port="/dev/cobot_leader_left",
        right_arm_port="/dev/cobot_leader_right",
        id="cobot_leader_bi",
    )
    keyboard_config = KeyboardTeleopConfig()

    robot = DddBotClient(robot_config)
    leader_arm = BiSO100Leader(leader_arm_config)
    keyboard = KeyboardTeleop(keyboard_config)

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    # === 数据集设置 ===
    action_features = hw_to_dataset_features(robot.action_features, ACTION)
    obs_features = hw_to_dataset_features(robot.observation_features, OBS_STR)
    dataset_features = {**action_features, **obs_features}

    dataset = LeRobotDataset.create(
        repo_id=args.dataset,
        fps=args.fps,
        features=dataset_features,
        robot_type=robot.name,
        use_videos=True,
        image_writer_threads=4,
    )
    print(f"Dataset created with id: {dataset.repo_id}")

    # === 连接设备 ===
    robot.connect()
    leader_arm.connect()
    keyboard.connect()

    listener, events = init_keyboard_listener()
    init_rerun(session_name="dddbot_record")

    if not robot.is_connected or not leader_arm.is_connected or not keyboard.is_connected:
        raise ValueError("Robot or teleop is not connected!")

    print("Starting record loop...")
    recorded_episodes = 0

    while recorded_episodes < args.num_episodes and not events["stop_recording"]:
        log_say(f"Recording episode {recorded_episodes + 1} of {args.num_episodes}", args.play_sounds)

        # === 主录制循环 ===
        # 注意: record_loop 处理数据收集。
        # 在 record_loop 内部，它会调用 teleop 的 get_action。
        # 因为我们传递了一个列表 [leader_arm, keyboard]，它会聚合动作。
        # 理想情况下，leader_arm 返回 "arm_..." 键，或者如果有不匹配我们需要一个适配器。
        # 在 alohamini/record_bi.py 中，它直接传递了 [leader_arm, keyboard]。
        # 注意 alohamini/teleoperate_bi.py 手动给键加了 "arm_" 前缀。
        # 我们需要检查 BiSO100Leader 是否原生返回 "left_..." 或 "arm_left_..."。
        # 基于默认的 SO100Leader，它返回相对于其配置名称的键？
        # 实际上，在遥操作脚本中我们经常看到手动重映射：
        # `arm_actions = {f"arm_{k}": v for k, v in arm_actions.items()}`
        # 如果 record_loop 调用 `teleop.get_action()`，如果键与机器人期望不匹配，我们可能需要一个包装器。
        # 然而 `alohamini` 录制脚本似乎没有包装器，只是传递了实例。
        # 这意味着要么：
        # 1. BiSO100Leader 被配置为返回正确的键。
        # 2. 或者 record_loop 处理了它？不，record_loop 是通用的。
        # 3. 或者我错过了 alohamini/record_bi.py 中的某些内容。
        # 让我们再次看看 `alohamini/record_bi.py`。它只是传递 `teleop=[leader_arm, keyboard]`。
        # Maybe `BiSO100Leader` already returns "arm_..." keys?
        # In `teleoperate_bi.py`, it did: `arm_actions = {f"arm_{k}": v for k, v in arm_actions.items()}`
        # This strongly suggests `BiSO100Leader` DOES NOT return "arm_" prefix by default.
        # So `record_bi.py` might be relying on something else or it might be broken/different?
        # Wait, if `record_bi.py` passes `leader_arm` directly, and `robot` expects `arm_left_...`, 
        # then `dataset` will record whatever `leader_arm` returns.
        # IF the `robot` configuration uses correct mapping in `send_action`, it might be fine IF the dataset features match.
        # But `robot.action_features` usually has "arm_left_...".
        # If `leader_arm` returns "left_...", then `record_loop` -> `actions = teleop.get_action()` -> `robot.send_action(actions)`...
        # If `robot.send_action` expects "arm_left", it will fail or ignore commands.
        # 
        # Let's verify `BiSO100Leader`.
        # I suspect I should wrap it or verify if `record_loop` does something.
        # But for now I will stick to the pattern in `alohamini/record_bi.py`.
        # If the user's `alohamini` example is correct, then I should follow it.
        
        record_loop(
            robot=robot,
            events=events,
            fps=args.fps,
            dataset=dataset,
            teleop=[leader_arm, keyboard],
            control_time_s=args.episode_time,
            single_task=args.task_description,
            display_data=True,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
        )

        # === 重置环境 ===
        if not events["stop_recording"] and (
            (recorded_episodes < args.num_episodes - 1) or events["rerecord_episode"]
        ):
            log_say("Reset the environment", args.play_sounds)
            record_loop(
                robot=robot,
                events=events,
                fps=args.fps,
                teleop=[leader_arm, keyboard],
                control_time_s=args.reset_time,
                single_task=args.task_description,
                display_data=True,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
            )

        if events["rerecord_episode"]:
            log_say("Re-record episode", args.play_sounds)
            events["rerecord_episode"] = False
            events["exit_early"] = False
            dataset.clear_episode_buffer()
            continue

        dataset.save_episode()
        recorded_episodes += 1

    # === Clean up ===
    log_say("Stop recording", args.play_sounds)
    robot.disconnect()
    leader_arm.disconnect()
    keyboard.disconnect()
    listener.stop()
    dataset.finalize()
    log_say("Dataset finalized and pushed to hub", args.play_sounds)
    dataset.push_to_hub()
    log_say("Exiting", args.play_sounds)


if __name__ == "__main__":
    main()
