#!/usr/bin/env python

import base64
import json
import logging
import time
from dataclasses import dataclass, field

import cv2
import draccus
import zmq

# Import DddBot configs and class
from .config_dddbot import DddBotConfig, LeKiwiHostConfig
from .dddbot import DddBot


@dataclass
class LeKiwiServerConfig:
    """Configuration for the Host script (serving DddBot)."""

    robot: DddBotConfig = field(default_factory=DddBotConfig)
    host: LeKiwiHostConfig = field(default_factory=LeKiwiHostConfig)


class LeKiwiHost:
    """
    Host Agent for DddBot.
    Note: Class name 'LeKiwiHost' retained as per requirements, but manages DddBot.
    """
    def __init__(self, config: LeKiwiHostConfig):
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


@draccus.wrap()
def main(cfg: LeKiwiServerConfig):
    logging.info("Configuring DddBot")
    # Initialize DddBot instead of LeKiwi
    robot = DddBot(cfg.robot)

    logging.info("Connecting DddBot")
    robot.connect()

    logging.info("Starting LeKiwiHost (for DddBot)")
    host = LeKiwiHost(cfg.host)

    last_cmd_time = time.time()
    watchdog_active = False
    logging.info("Waiting for commands...")
    try:
        # Business logic
        start = time.perf_counter()
        duration = 0
        while duration < host.connection_time_s:
            loop_start_time = time.time()
            try:
                msg = host.zmq_cmd_socket.recv_string(zmq.NOBLOCK)
                data = dict(json.loads(msg))
                _action_sent = robot.send_action(data)
                last_cmd_time = time.time()
                watchdog_active = False
            except zmq.Again:
                if not watchdog_active:
                    logging.warning("No command available")
            except Exception as e:
                logging.error("Message fetching failed: %s", e)

            now = time.time()
            if (now - last_cmd_time > host.watchdog_timeout_ms / 1000) and not watchdog_active:
                logging.warning(
                    f"Command not received for more than {host.watchdog_timeout_ms} milliseconds. Stopping the base."
                )
                watchdog_active = True
                robot.stop_base()

            last_observation = robot.get_observation()

            # Encode ndarrays to base64 strings
            for cam_key, _ in robot.cameras.items():
                if cam_key in last_observation:
                    ret, buffer = cv2.imencode(
                        ".jpg", last_observation[cam_key], [int(cv2.IMWRITE_JPEG_QUALITY), 90]
                    )
                    if ret:
                        last_observation[cam_key] = base64.b64encode(buffer).decode("utf-8")
                    else:
                        last_observation[cam_key] = ""

            # Send the observation to the remote agent
            try:
                host.zmq_observation_socket.send_string(json.dumps(last_observation), flags=zmq.NOBLOCK)
            except zmq.Again:
                logging.info("Dropping observation, no client connected")

            # Ensure a short sleep to avoid overloading the CPU.
            elapsed = time.time() - loop_start_time

            time.sleep(max(1 / host.max_loop_freq_hz - elapsed, 0))
            duration = time.perf_counter() - start
        print("Cycle time reached.")

    except KeyboardInterrupt:
        print("Keyboard interrupt received. Exiting...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("Shutting down Host.")
        try:
            robot.disconnect()
        except:
            pass
        try:
            host.disconnect()
        except:
            pass

    logging.info("Finished cleanly")


if __name__ == "__main__":
    main()
