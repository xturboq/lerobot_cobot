#!/usr/bin/env python

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

import base64
import json
import logging
import time

import cv2
import zmq

from .config_cobot import CobotConfig, CobotHostConfig
from .cobot import Cobot


class CobotHost:
    """Cobot Host服务 - 运行在机器人端"""

    def __init__(self, config: CobotHostConfig):
        self.zmq_context = zmq.Context()
        self.zmq_cmd_socket = self.zmq_context.socket(zmq.PULL)
        self.zmq_cmd_socket.setsockopt(zmq.CONFLATE, 1)
        self.zmq_cmd_socket.bind(f"tcp://*:{config.port_zmq_cmd}")

        self.zmq_observation_socket = self.zmq_context.socket(zmq.PUSH)
        self.zmq_observation_socket.setsockopt(zmq.CONFLATE, 1)
        self.zmq_observation_socket.bind(f"tcp://*:{config.port_zmq_observations}")

        self.connection_time_s = config.connection_time_s
        self.watchdog_timeout_ms = config.watchdog_timeout_ms
        self.max_loop_freq_hz = config.max_loop_freq_hz

    def disconnect(self):
        self.zmq_observation_socket.close()
        self.zmq_cmd_socket.close()
        self.zmq_context.term()


def main():
    logging.basicConfig(level=logging.INFO)
    logging.info("Configuring Cobot")
    robot_config = CobotConfig()
    robot_config.id = "CobotRobot"
    robot = Cobot(robot_config)

    logging.info("Connecting Cobot")
    robot.connect()

    logging.info("Starting Cobot HostAgent")
    host_config = CobotHostConfig()
    host = CobotHost(host_config)

    last_cmd_time = time.time()
    watchdog_active = False
    logging.info("Waiting for commands...")

    try:
        # 主循环
        start = time.perf_counter()
        duration = 0

        while duration < host.connection_time_s:
            loop_start_time = time.time()
            
            # 接收命令
            try:
                msg = host.zmq_cmd_socket.recv_string(zmq.NOBLOCK)
                data = dict(json.loads(msg))
                _action_sent = robot.send_action(data)

                last_cmd_time = time.time()
                watchdog_active = False
            except zmq.Again:
                if not watchdog_active:
                    logging.debug("No command available")
            except Exception as e:
                logging.exception("Message fetching failed: %s", e)

            # 看门狗：如果长时间没收到命令，停止底盘
            now = time.time()
            if (now - last_cmd_time > host.watchdog_timeout_ms / 1000) and not watchdog_active:
                logging.warning(
                    f"Command not received for more than {host.watchdog_timeout_ms} milliseconds. Stopping the base."
                )
                watchdog_active = True
                robot.stop_base()

            # 获取观测数据
            last_observation = robot.get_observation()

            # 将图像编码为base64字符串
            for cam_key, _ in robot.cameras.items():
                ret, buffer = cv2.imencode(
                    ".jpg", last_observation[cam_key], [int(cv2.IMWRITE_JPEG_QUALITY), 90]
                )
                if ret:
                    last_observation[cam_key] = base64.b64encode(buffer).decode("utf-8")
                else:
                    last_observation[cam_key] = ""

            # 发送观测数据到远程PC
            try:
                host.zmq_observation_socket.send_string(json.dumps(last_observation), flags=zmq.NOBLOCK)
            except zmq.Again:
                logging.debug("Dropping observation, no client connected")

            # 控制循环频率
            elapsed = time.time() - loop_start_time
            time.sleep(max(1 / host.max_loop_freq_hz - elapsed, 0))
            duration = time.perf_counter() - start
        
        print("Connection time reached.")

    except KeyboardInterrupt:
        print("Keyboard interrupt received. Exiting...")
    finally:
        print("Shutting down Cobot Host.")
        robot.disconnect()
        host.disconnect()

    logging.info("Finished Cobot Host cleanly")


if __name__ == "__main__":
    main()

