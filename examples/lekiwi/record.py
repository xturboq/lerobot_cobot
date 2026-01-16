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
from lerobot.robots.lekiwi.config_lekiwi import LeKiwiClientConfig
from lerobot.robots.lekiwi.lekiwi_client import LeKiwiClient
from lerobot.scripts.lerobot_record import record_loop
from lerobot.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.teleoperators.bi_so100_leader import BiSO100Leader, BiSO100LeaderConfig
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say, init_logging
from lerobot.utils.visualization_utils import init_rerun

NUM_EPISODES = 10
FPS = 30
EPISODE_TIME_SEC = 60
RESET_TIME_SEC = 30
TASK_DESCRIPTION = "Grab the purple cube"
HF_REPO_ID = "coola/011301"
DISPLAY_DATA = True
PUSH_TO_HUB = False


def main():
    init_logging()
    
    # Create the robot and teleoperator configurations
    robot_config = LeKiwiClientConfig(remote_ip="127.0.0.1", id="lekiwi")
    leader_arm_config = BiSO100LeaderConfig(
        left_arm_port="/dev/cobot_leader_left",
        right_arm_port="/dev/cobot_leader_right",
        id="my_bi_leader_arm"
    )
    keyboard_config = KeyboardTeleopConfig()

    # Initialize the robot and teleoperator
    robot = LeKiwiClient(robot_config)
    leader_arm = BiSO100Leader(leader_arm_config)
    keyboard = KeyboardTeleop(keyboard_config)

    # TODO(Steven): Update this example to use pipelines
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    # Configure the dataset features
    action_features = hw_to_dataset_features(robot.action_features, ACTION)
    obs_features = hw_to_dataset_features(robot.observation_features, OBS_STR)
    dataset_features = {**action_features, **obs_features}

    # Create the dataset
    dataset = LeRobotDataset.create(
        repo_id=HF_REPO_ID,
        fps=FPS,
        features=dataset_features,
        robot_type=robot.name,
        use_videos=True,
        image_writer_threads=4,
    )

    # Connect the robot and teleoperator
    # To connect you already should have this script running on LeKiwi: `python -m lerobot.robots.lekiwi.lekiwi_host --robot.id=my_awesome_kiwi`
    robot.connect()
    leader_arm.connect()
    keyboard.connect()

    # Initialize the keyboard listener and rerun visualization
    listener, events = init_keyboard_listener()
    if DISPLAY_DATA:
        init_rerun(session_name="lekiwi_record")

    if not robot.is_connected or not leader_arm.is_connected or not keyboard.is_connected:
        raise ValueError("Robot or teleop is not connected!")

    log_say("All devices connected. Ready to record.")
    print("\n" + "="*50)
    print("Keyboard Controls:")
    print("  [Enter]  - End current episode early")
    print("  [Backspace] - Re-record current episode")
    print("  [Escape] - Stop recording completely")
    print("="*50 + "\n", flush=True)

    log_say("Starting record loop...")
    recorded_episodes = 0
    while recorded_episodes < NUM_EPISODES and not events["stop_recording"]:
        print(f"\n>>> Recording episode {recorded_episodes + 1} of {NUM_EPISODES} ({EPISODE_TIME_SEC}s) <<<", flush=True)
        log_say(f"Recording episode {recorded_episodes + 1} of {NUM_EPISODES}")

        # Main record loop
        record_loop(
            robot=robot,
            events=events,
            fps=FPS,
            dataset=dataset,
            teleop=[leader_arm, keyboard],
            control_time_s=EPISODE_TIME_SEC,
            single_task=TASK_DESCRIPTION,
            display_data=DISPLAY_DATA,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
        )

        # Reset the environment if not stopping or re-recording
        if not events["stop_recording"] and (
            (recorded_episodes < NUM_EPISODES - 1) or events["rerecord_episode"]
        ):
            # Clear exit_early so reset loop can run fully
            events["exit_early"] = False
            print(f"\n>>> RESET ENVIRONMENT ({RESET_TIME_SEC}s) - Move robot to starting position <<<", flush=True)
            log_say(f"Reset the environment. You have {RESET_TIME_SEC} seconds.")
            record_loop(
                robot=robot,
                events=events,
                fps=FPS,
                teleop=[leader_arm, keyboard],
                control_time_s=RESET_TIME_SEC,
                single_task=TASK_DESCRIPTION,
                display_data=DISPLAY_DATA,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
            )

        if events["rerecord_episode"]:
            print(f"\n>>> RE-RECORDING episode {recorded_episodes + 1} <<<", flush=True)
            log_say(f"Re-recording episode {recorded_episodes + 1}")
            events["rerecord_episode"] = False
            events["exit_early"] = False
            dataset.clear_episode_buffer()
            continue

        # Save episode
        print(f"\n>>> Saving episode {recorded_episodes + 1}... <<<", flush=True)
        log_say(f"Saving episode {recorded_episodes + 1}")
        dataset.save_episode()
        recorded_episodes += 1
        print(f">>> Episode {recorded_episodes} saved successfully! <<<\n", flush=True)
        log_say(f"Episode {recorded_episodes} saved successfully")

    # Clean up
    log_say("Stop recording")
    robot.disconnect()
    leader_arm.disconnect()
    keyboard.disconnect()
    listener.stop()

    log_say("Finalizing dataset...")
    dataset.finalize()
    if PUSH_TO_HUB:
        log_say("Pushing dataset to hub...")
        dataset.push_to_hub()

    log_say(f"Recording complete! {recorded_episodes} episodes saved to {HF_REPO_ID}")


if __name__ == "__main__":
    main()
