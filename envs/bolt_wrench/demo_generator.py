import argparse
import datetime
import io
import os
import sys
import time
import subprocess
import socket
from typing import Optional
from xml.etree import ElementTree as ET

import numpy as np
from ruamel.yaml import YAML

from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        type=str,
        default=os.path.join(os.path.dirname(os.path.realpath(__file__)), "rsc", "config.yaml"),
    )
    parser.add_argument("--steps", type=int, default=-1, help="number of control steps (-1: until Ctrl+C)")
    parser.add_argument("--sleep_scale", type=float, default=1.0, help="wall-time slowdown factor")
    parser.add_argument("--pre_reset_sleep", type=float, default=0.0, help="seconds to wait before first reset")
    parser.add_argument("--episode_steps", type=int, default=0, help="steps per episode (0: use config max_total_steps, <0: no reset)")
    parser.add_argument("--render", action="store_true", help="force visualization on")
    args = parser.parse_args()

    task_path = os.path.dirname(os.path.realpath(__file__))
    build_path = os.path.join(task_path, "build")
    sys.path.append(build_path)

    import bolt_wrench

    cfg = YAML().load(open(args.cfg, "r"))
    cfg["environment"]["use_action_command"] = False

    def find_available_port(base_port: int, max_tries: int = 10) -> Optional[int]:
        for i in range(max_tries):
            port = base_port + i
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("0.0.0.0", port))
                return port
            except OSError:
                continue
            finally:
                sock.close()
        return None

    def update_gui_settings(gui_path: str, port: int) -> bool:
        try:
            tree = ET.parse(gui_path)
            root = tree.getroot()
            node = root.find("./ip_port")
            if node is None:
                print(f"[unity] ip_port not found in {gui_path}")
                return False
            node.set("value", str(port))
            tree.write(gui_path)
            return True
        except Exception as exc:
            print(f"[unity] failed to update gui_settings.xml: {exc}")
            return False

    def maybe_launch_unity(cfg_env: dict) -> Optional[subprocess.Popen]:
        if not cfg_env.get("launch_unity", False):
            return None
        unity_bin = cfg_env.get(
            "unity_bin",
            "/home/Jiseong/raisim_ws/raisimlib/raisimUnity/linux/raisimUnity.x86_64",
        )
        unity_gui = cfg_env.get(
            "unity_gui_settings",
            "/home/Jiseong/raisim_ws/raisimlib/raisimUnity/linux/gui_settings.xml",
        )
        if not os.path.isfile(unity_bin) or not os.access(unity_bin, os.X_OK):
            print(f"[unity] binary not found or not executable: {unity_bin}")
            return None

        base_port = int(cfg_env.get("render_port", 8080))
        port = find_available_port(base_port)
        if port is None:
            print("[unity] no available port for RaisimServer; skipping Unity launch")
            return None
        cfg_env["render_port"] = port
        if not update_gui_settings(unity_gui, port):
            print("[unity] failed to update gui_settings.xml; skipping Unity launch")
            return None
        print(f"[unity] launching RaisimUnity on port {port}")
        return subprocess.Popen([unity_bin])

    unity_proc = None
    if args.render or cfg["environment"].get("render", False):
        unity_proc = maybe_launch_unity(cfg["environment"])

    yaml_dumper = YAML()
    string_stream = io.StringIO()
    yaml_dumper.dump(cfg["environment"], string_stream)
    cfg_string = string_stream.getvalue()

    env = VecEnv(bolt_wrench.RaisimGymEnv(task_path + "/rsc", cfg_string))
    if args.pre_reset_sleep > 0.0:
        time.sleep(args.pre_reset_sleep)
    env.reset()

    if args.render or cfg["environment"].get("render", False):
        env.turn_on_visualization()

    action = np.zeros((env.num_envs, env.num_acts), dtype=np.float32)

    def init_logs():
        return [], [], [], [], [], []

    log_bolt_gc, log_bolt_gv, log_wrench_gc, log_wrench_gv, log_robot_gc, log_robot_gv = init_logs()

    control_dt = float(cfg["environment"].get("control_dt", 0.01))
    sleep_sec = max(0.0, control_dt * args.sleep_scale)

    def save_logs():
        if not log_bolt_gc:
            return
        time_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        out_dir = os.path.join(task_path, "data", "demo", time_str)
        os.makedirs(out_dir, exist_ok=True)

        np.save(os.path.join(out_dir, "demo_bolt_gc.npy"), np.asarray(log_bolt_gc, dtype=np.float64))
        np.save(os.path.join(out_dir, "demo_bolt_gv.npy"), np.asarray(log_bolt_gv, dtype=np.float64))
        np.save(os.path.join(out_dir, "demo_wrench_gc.npy"), np.asarray(log_wrench_gc, dtype=np.float64))
        np.save(os.path.join(out_dir, "demo_wrench_gv.npy"), np.asarray(log_wrench_gv, dtype=np.float64))
        np.save(os.path.join(out_dir, "demo_robot_gc.npy"), np.asarray(log_robot_gc, dtype=np.float64))
        np.save(os.path.join(out_dir, "demo_robot_gv.npy"), np.asarray(log_robot_gv, dtype=np.float64))
        print(f"Saved demo data to {out_dir}")

    step = 0
    sim_time = 0.0
    reset_time = 100.0
    try:
        while args.steps < 0 or step < args.steps:
            step_start = time.perf_counter()
            env.step(action)
            bolt_gc, bolt_gv, wrench_gc, wrench_gv, robot_gc, robot_gv = env.wrapper.get_demo_state()
            log_bolt_gc.append(bolt_gc[0].copy())
            log_bolt_gv.append(bolt_gv[0].copy())
            log_wrench_gc.append(wrench_gc[0].copy())
            log_wrench_gv.append(wrench_gv[0].copy())
            log_robot_gc.append(robot_gc[0].copy())
            log_robot_gv.append(robot_gv[0].copy())
            step += 1
            sim_time += control_dt
            if sim_time >= reset_time:
                save_logs()
                log_bolt_gc, log_bolt_gv, log_wrench_gc, log_wrench_gv, log_robot_gc, log_robot_gv = init_logs()
                env.reset()
                sim_time = 0.0
            if sleep_sec > 0.0:
                elapsed = time.perf_counter() - step_start
                remaining = sleep_sec - elapsed
                if remaining > 0.0:
                    time.sleep(remaining)
    except KeyboardInterrupt:
        pass

    env.turn_off_visualization()
    env.close()
    if unity_proc is not None:
        unity_proc.terminate()

    save_logs()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
