import argparse
import io
import os
import sys

from ruamel.yaml import YAML

from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv


def main() -> int:
    parser = argparse.ArgumentParser(description="FT vs PD dt/cutoff sweep")
    parser.add_argument(
        "--cfg",
        type=str,
        default=os.path.join(os.path.dirname(os.path.realpath(__file__)), "rsc", "config.yaml"),
    )
    parser.add_argument("--render", action="store_true", help="enable visualization for env 0")
    parser.add_argument("--dt_min", type=float, default=0.0010)
    parser.add_argument("--dt_max", type=float, default=0.0025)
    parser.add_argument("--dt_step", type=float, default=0.0001)
    parser.add_argument("--cutoff_min", type=float, default=5.0)
    parser.add_argument("--cutoff_max", type=float, default=50.0)
    parser.add_argument("--cutoff_step", type=float, default=1.0)
    parser.add_argument("--sample_time", type=float, default=2.5)
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument(
        "--fixed_dt_step",
        action="store_true",
        help="disable dt randomization (default: randomize dt each step)",
    )
    parser.add_argument("--urdf_type", type=str, default="_stub-0")
    args = parser.parse_args()

    task_path = os.path.dirname(os.path.realpath(__file__))
    build_path = os.path.join(task_path, "build")
    sys.path.append(build_path)

    import bolt_wrench

    cfg = YAML().load(open(args.cfg, "r"))
    cfg_env = cfg["environment"]
    cfg_env["use_action_command"] = False
    cfg_env["num_envs"] = 1
    cfg_env["num_threads"] = 1
    cfg_env["render"] = bool(args.render)

    yaml_dumper = YAML()
    string_stream = io.StringIO()
    yaml_dumper.dump(cfg_env, string_stream)
    cfg_string = string_stream.getvalue()

    env = VecEnv(bolt_wrench.RaisimGymEnv(task_path + "/rsc", cfg_string))
    env.reset()
    if args.render:
        env.turn_on_visualization()

    results = env.wrapper.run_ft_pd_sweep(
        args.dt_min,
        args.dt_max,
        args.dt_step,
        args.cutoff_min,
        args.cutoff_max,
        args.cutoff_step,
        args.sample_time,
        int(args.num_samples),
        not bool(args.fixed_dt_step),
        args.urdf_type,
    )

    print(
        f"# FT vs PD sweep complete: {len(results)} cases "
        f"(dt: {args.dt_min}..{args.dt_max}, cutoff: {args.cutoff_min}..{args.cutoff_max})"
    )
    print("case,dt_s,cutoff_hz,num_fail,num_samples,fail_rate")
    for i, (dt_s, cutoff_hz, num_fail, num_samples) in enumerate(results, start=1):
        fail_rate = (num_fail / num_samples) if num_samples > 0 else 0.0
        print(f"{i},{dt_s:.7f},{cutoff_hz:.2f},{int(num_fail)},{int(num_samples)},{fail_rate:.3f}")

    env.turn_off_visualization()
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
