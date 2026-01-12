import argparse
import time

from lerobot.robots.dddbot.dddbot_client import DddBotClient, DddBotClientConfig
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.teleoperators.bi_so100_leader import BiSO100Leader, BiSO100LeaderConfig
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

# ============ 参数部分 ============ #
parser = argparse.ArgumentParser()
parser.add_argument("--use_dummy", action="store_true", help="不连接机器人，仅打印动作")
parser.add_argument("--fps", type=int, default=30, help="主循环频率（每秒帧数）")
parser.add_argument("--remote_ip", type=str, default="127.0.0.1", help="DDDBot 主机 IP 地址")
parser.add_argument("--robot_id", type=str, default="dddbot", help="机器人 ID")

args = parser.parse_args()

USE_DUMMY = args.use_dummy
FPS = args.fps
# ========================================== #

if USE_DUMMY:
    print("🧪 已启用 USE_DUMMY 模式：机器人将不会连接，仅打印动作。")

# 创建配置
robot_config = DddBotClientConfig(remote_ip=args.remote_ip, id=args.robot_id)
bi_cfg = BiSO100LeaderConfig(
    left_arm_port="/dev/cobot_leader_left",
    right_arm_port="/dev/cobot_leader_right",
    id="cobot_leader_bi",
)
leader = BiSO100Leader(bi_cfg)
keyboard_config = KeyboardTeleopConfig(id="my_laptop_keyboard")
keyboard = KeyboardTeleop(keyboard_config)
robot = DddBotClient(robot_config)

# 连接逻辑
if not USE_DUMMY:
    robot.connect()
else:
    print("🧪 跳过 robot.connect()，仅打印动作。")

leader.connect()
keyboard.connect()

init_rerun(session_name="dddbot_teleop")

if not robot.is_connected or not leader.is_connected or not keyboard.is_connected:
    print("⚠️ 警告：部分设备未连接！仍在运行以便调试。")

print("开始遥操作循环...")

# 主循环
while True:
    t0 = time.perf_counter()

    observation = robot.get_observation() if not USE_DUMMY else {}

    # 获取主臂动作
    arm_actions = leader.get_action()
    
    # DddBot / Cobot 双臂格式通常期望：
    # arm_left_...
    # arm_right_...
    # BiSO100Leader 返回像 "left_name", "right_name" 这样的键。
    # 我们需要将它们映射到 "arm_left_name", "arm_right_name"。
    # 让我们检查 BiSO100Leader 输出格式。
    # 通常它返回 { "left_shoulder_pan": val, ... "right_shoulder_pan": val ... }
    # 但是 DddBotClient 期望动作字典中有 "arm_left_shoulder_pan.pos" 等？
    # DddBotClient.send_action 基于 _state_order 将动作字典转换为列表。
    # _state_order 拥有 "arm_left_shoulder_pan.pos"。
    # Leader 通常返回 "name"。
    # 让我们看看 `alohamini/teleoperate_bi.py`：
    # arm_actions = {f"arm_{k}": v for k, v in arm_actions.items()}
    # 这暗示 leader 返回 "left_..." 而 aloha 期望 "arm_left_..."
    
    mapped_arm_actions = {f"arm_{k}": v for k, v in arm_actions.items()}

    keyboard_keys = keyboard.get_action()
    base_action = robot._from_keyboard_to_base_action(keyboard_keys)
    
    # 合并动作
    action = {**mapped_arm_actions, **base_action}
    
    log_rerun_data(observation, action)

    if USE_DUMMY:
        print(f"[USE_DUMMY] action → {action}")
    else:
        robot.send_action(action)
        # print(f"Sent action → {action}")

    precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
