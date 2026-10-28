#!/usr/bin/env python
"""
Local job queue for the BPO paper reproduction (see REPRODUCTION_NOTES.md).

Builds every run from the repo's sweep files, plus the Fig. 9 ablations (which have
no sweep files), and runs them with a concurrency limit per pool. Results go to wandb;
job status is kept in <out>/jobs-<prefix>.json, so the launcher can be stopped and
restarted without redoing finished runs.

    python scripts/repro_launcher.py list
    python scripts/repro_launcher.py run --dry-run
    python scripts/repro_launcher.py run
    # short calibration runs into a separate wandb project and state file:
    python scripts/repro_launcher.py run --prefix BPO-calib --max-steps 200000 --max-iters 30 --only g1_rough-bpo-s1
"""
import argparse
import itertools
import json
import os
import re
import signal
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import psutil
import yaml

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable
ABLATION_SEEDS = [1, 2, 3, 4, 5]
# BPO/PPO Atari lengths follow the plotted horizons in Fig. 4. DQN, SAC and MuJoCo
# use the 40M default in cfgs/config.yaml.
ATARI_STEPS = {"AsterixNoFrameskip-v4": 40_000_000}
ATARI_DEFAULT_STEPS = 20_000_000


@dataclass
class Job:
    name: str
    group: str
    pool: str  # "cpu", "atari" or "isaac"
    kind: str  # "sb3" or "isaac"
    overrides: dict
    args: list = field(default_factory=list)  # fixed CLI args (isaac only)
    gpu: bool = False
    after: str = None  # resume jobs: the job whose saved model this one continues


def fmt(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def grid(sweep):
    spec = yaml.safe_load(open(REPO / sweep))
    params = spec["parameters"]
    keys = list(params)
    for combo in itertools.product(*(params[k]["values"] for k in keys)):
        yield dict(zip(keys, combo)), spec


def sb3_jobs(sweep, label, pool, extra=None, gpu=False):
    for p, _ in grid(sweep):
        ovr = {**p, **(extra or {})}
        if "algo.seed" not in ovr and label in ("bpo", "ppo", "bpo_lam0.001"):
            ovr["++algo.seed"] = ovr["training.seed"]  # seed the networks too, not only the env
        if ovr.get("env") == "atari" and label in ("bpo", "ppo") and "training.total_timesteps" not in ovr:
            ovr["training.total_timesteps"] = ATARI_STEPS.get(ovr["env.env_id"], ATARI_DEFAULT_STEPS)
        group = f"{ovr['env.env_id']}-{label}"
        yield Job(f"{group}-s{ovr['training.seed']}", group, pool, "sb3", ovr, gpu=gpu)


def isaac_jobs(sweep, algorithm, label, extra=None):
    task_short = Path(sweep).parent.name
    for p, spec in grid(sweep):
        ovr = {**p, **(extra or {})}
        fixed = [c for c in spec["command"] if c.startswith("--")]
        group = f"{task_short}-{label}"
        yield Job(f"{group}-s{ovr['agent.seed']}", group, "isaac", "isaac", ovr,
                  args=["--algorithm", algorithm, *fixed], gpu=True)


def build_jobs(sac_device, atari_device, max_hours):
    jobs = []
    # IsaacLab, Fig. 5 (5 seeds each). Longest tasks first.
    for task in ["g1_rough", "h1_rough", "anymal_c_rough", "go1_rough"]:
        jobs += isaac_jobs(f"sweeps/isaaclab/{task}/bpo_seed.yaml", "bpo", "bpo")
        jobs += isaac_jobs(f"sweeps/isaaclab/{task}/ppo_seed.yaml", "ppo", "ppo")
    # Fig. 9 ablations on G1-rough. BPO and PPO reuse the Fig. 5 runs.
    jobs += isaac_jobs("sweeps/isaaclab/g1_rough/bpo_value_seed.yaml", "bpo", "bpo_value")
    g1 = "sweeps/isaaclab/g1_rough/bpo_seed.yaml"
    ablations = {
        "bpo_tv": {"agent.algorithm.loss_type": "TV"},
        "bpo_rkl": {"agent.algorithm.loss_type": "RKL"},
        **{f"bpo_lam{v}": {"agent.algorithm.reg_w": v} for v in [0.01, 0.03, 0.1]},
        **{f"bpo_tvcoef{v}": {"agent.algorithm.tv_loss_coef": v} for v in [0.1, 0.3, 1.0, 3.0, 10.0]},
    }
    for label, extra in ablations.items():
        jobs += [j for j in isaac_jobs(g1, "bpo", label, extra) if j.overrides["agent.seed"] in ABLATION_SEEDS]

    # Atari: BPO, PPO, DQN. Asterix first, it runs twice as long.
    on_gpu = atari_device != "cpu"
    pool, dev = ("atari", {}) if on_gpu else ("cpu", {"++algo.device": "cpu"})
    atari = []
    for algo in ["bpo", "ppo", "dqn"]:
        atari += sb3_jobs(f"sweeps/atari/{algo}_seed.yaml", algo, pool, dev, gpu=on_gpu)
    jobs += sorted(atari, key=lambda j: "Asterix" not in j.name)

    # MuJoCo (CPU). SAC first, it is the slowest; then the longest on-policy envs.
    sac_extra = {"env": "mujoco", "++algo.device": sac_device}
    for env in ["Humanoid", "Ant", "Hopper", "Swimmer"]:
        jobs += sb3_jobs(f"sweeps/mujoco/{env}-V4/sac_seed.yaml", "sac",
                         "cpu" if sac_device == "cpu" else "atari", sac_extra, gpu=sac_device != "cpu")
    for env in ["Humanoid", "Hopper", "Ant", "Swimmer"]:
        for algo in ["bpo", "ppo"]:
            jobs += sb3_jobs(f"sweeps/mujoco/{env}-V4/{algo}_seed.yaml", algo, "cpu", {"++algo.device": "cpu"})
    jobs += sb3_jobs("sweeps/mujoco/Hopper-V4/bpo_seed_lambda_paper.yaml", "bpo_lam0.001", "cpu",
                     {"++algo.device": "cpu"})
    # Wall-clock caps, e.g. sac=24: the authors' cluster jobs were limited to 12-24 h.
    for j in jobs:
        label = j.group.rsplit("-", 1)[-1]
        if j.kind == "sb3" and label in max_hours:
            j.overrides["training.max_hours"] = max_hours[label]
    return jobs


def command(job, prefix, max_steps, max_iters):
    if job.kind == "sb3":
        ovr = dict(job.overrides)
        if max_steps:
            ovr["training.total_timesteps"] = max_steps
        domain = "mujoco" if ovr.get("env") == "mujoco" else "atari"
        args = [f"{k}={fmt(v)}" for k, v in ovr.items()]
        # a resume continues the original wandb run under its name
        args += [f"wandb.project={prefix}-{domain}", f"++wandb.name={job.after or job.name}",
                 f"++wandb.group={job.group}", f"hydra.run.dir=outputs/{job.name}"]
        return [PY, str(REPO / "train.py"), *args]
    cmd = [PY, "train_rsl_rl.py", *job.args, "--log_project_name", f"{prefix}-isaaclab",
           "--run_name", job.name]
    if max_iters:
        cmd += ["--max_iterations", str(max_iters)]
    return cmd + ["agent.save_interval=500"] + [f"{k}={fmt(v)}" for k, v in job.overrides.items()]


def job_env(job, gpu):
    # OSMesa renders MuJoCo videos on the CPU; EGL would hold ~160 MB of GPU 0 per process
    env = dict(os.environ, OMNI_KIT_ACCEPT_EULA="YES", MUJOCO_GL="osmesa", PYOPENGL_PLATFORM="osmesa",
               WANDB_SILENT="true")
    if job.kind == "sb3":
        env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    env["CUDA_VISIBLE_DEVICES"] = str(gpu) if gpu is not None else ""
    return env


def pick_gpu(job, running, limits, gpus):
    """GPU index (or None for CPU jobs) if a slot is free in the job's pool, else False."""
    in_pool = [r for r in running.values() if r["job"].pool == job.pool]
    if not job.gpu:
        return None if len(in_pool) < limits[job.pool] else False
    load = {g: sum(r["gpu"] == g for r in in_pool) for g in gpus}
    g = min(gpus, key=lambda g: load[g])
    return g if load[g] < limits[job.pool] else False


def find_pid(job):
    """PID of a live process running `job`, started by an earlier launcher, or None."""
    marker = ["--run_name", job.name] if job.kind == "isaac" else [f"hydra.run.dir=outputs/{job.name}"]
    for p in psutil.process_iter(["cmdline"]):
        cmd = p.info["cmdline"] or []
        if any(cmd[i:i + len(marker)] == marker for i in range(len(cmd))):
            return p.pid
    return None


def poll(r, log_path):
    """Exit code, or None while running. An adopted process isn't our child, so its exit code
    is unknown; it counts as failed if its log has a traceback."""
    if r["proc"] is not None:
        return r["proc"].poll()
    try:
        if psutil.Process(r["pid"]).status() != psutil.STATUS_ZOMBIE:
            return None
    except psutil.NoSuchProcess:
        pass
    return 1 if "Traceback" in log_path.read_text(errors="replace")[-20000:] else 0


def wandb_run_id(out, wandb_name):
    """Id of the SB3 wandb run named `wandb_name` that has a saved model, or None."""
    for meta in sorted((out / "sb3" / "wandb").glob("run-*/files/wandb-metadata.json"), reverse=True):
        if f"++wandb.name={wandb_name}" in json.loads(meta.read_text()).get("args", []):
            run_id = meta.parent.parent.name.rsplit("-", 1)[-1]
            if (out / "sb3" / "models" / run_id / "model.zip").exists():
                return run_id
    return None


def saved_steps(out, run_id):
    with zipfile.ZipFile(out / "sb3" / "models" / run_id / "model.zip") as z:
        return json.loads(z.read("data"))["num_timesteps"]


def run(jobs, args):
    out = Path(args.out)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    (out / "sb3").mkdir(exist_ok=True)
    state_file = out / f"jobs-{args.prefix}.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else {}
    skip = {"done"} if args.retry_failed else {"done", "failed"}
    pending = [j for j in jobs if state.get(j.name, {}).get("status") not in skip]
    limits = {"cpu": args.cpu, "atari": args.atari_per_gpu, "isaac": args.isaac_per_gpu}
    gpus = [int(g) for g in args.gpus.split(",")]

    # Adopt jobs still running from an earlier launcher (stopped with SIGKILL, which leaves them alive)
    running = {}
    for job in pending:
        if state.get(job.name, {}).get("status") == "running" and (pid := find_pid(job)):
            running[job.name] = {"proc": None, "pid": pid, "job": job, "gpu": state[job.name].get("gpu"),
                                 "t0": psutil.Process(pid).create_time(), "log": None}
    pending = [j for j in pending if j.name not in running]
    print(f"{len(pending)} jobs to run, {len(running)} adopted "
          f"({len(jobs) - len(pending) - len(running)} already done/failed)", flush=True)

    if args.dry_run:
        for j in pending:
            print(j.pool, " ".join(command(j, args.prefix, args.max_steps, args.max_iters)))
        return

    def stop(*_):
        for r in running.values():
            try:
                os.killpg(r["pid"], signal.SIGTERM)
            except ProcessLookupError:
                pass
        # Isaac Sim ignores SIGTERM, so escalate after a grace period
        deadline = time.time() + 30
        alive = lambda: [r for n, r in running.items() if poll(r, out / "logs" / f"{n}.log") is None]
        while time.time() < deadline and alive():
            time.sleep(1)
        for r in alive():
            try:
                os.killpg(r["pid"], signal.SIGKILL)
            except ProcessLookupError:
                pass
        for name in running:
            state[name] = {"status": "interrupted"}
        state_file.write_text(json.dumps(state, indent=1))
        print("stopped; interrupted jobs will rerun on the next start", flush=True)
        sys.exit(1)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    while pending or running:
        for name, r in list(running.items()):
            rc = poll(r, out / "logs" / f"{name}.log")
            if rc is None:
                continue
            if r["log"] is not None:
                r["log"].close()
            hours = round((time.time() - r["t0"]) / 3600, 2)
            state[name] = {"status": "done" if rc == 0 else "failed", "rc": rc, "hours": hours}
            state_file.write_text(json.dumps(state, indent=1))
            print(f"[{time.strftime('%m-%d %H:%M')}] {state[name]['status']:6s} {name} ({hours} h)", flush=True)
            del running[name]
        for job in list(pending):
            if job.after:
                before = state.get(job.after, {}).get("status")
                if before not in ("done", "failed"):
                    continue
                run_id = wandb_run_id(out, job.after) if before == "done" else None
                target = job.overrides["training.total_timesteps"]
                if run_id is None or saved_steps(out, run_id) >= target:
                    reason = "original failed" if before == "failed" else "no saved model"
                    state[job.name] = ({"status": "failed", "reason": reason} if run_id is None
                                       else {"status": "done", "reason": "already at target"})
                    pending.remove(job)
                    print(f"[{time.strftime('%m-%d %H:%M')}] skip   {job.name} ({state[job.name]['reason']})",
                          flush=True)
                    continue
            gpu = pick_gpu(job, running, limits, gpus)
            if gpu is False:
                continue
            if job.after:
                job.overrides["training.resume"] = run_id
            log = open(out / "logs" / f"{job.name}.log", "w")
            cwd = out / "sb3" if job.kind == "sb3" else REPO
            proc = subprocess.Popen(command(job, args.prefix, args.max_steps, args.max_iters), cwd=cwd,
                                    env=job_env(job, gpu), stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            running[job.name] = {"proc": proc, "pid": proc.pid, "job": job, "gpu": gpu, "t0": time.time(),
                                 "log": log}
            state[job.name] = {"status": "running", "gpu": gpu}
            pending.remove(job)
            print(f"[{time.strftime('%m-%d %H:%M')}] start  {job.name} (gpu {gpu})", flush=True)
            time.sleep(args.stagger)
        state_file.write_text(json.dumps(state, indent=1))
        time.sleep(10)
    print("all jobs finished", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["list", "run"])
    parser.add_argument("--out", default="/home2/hua48722/bpo_repro", help="logs, job state and SB3 outputs")
    parser.add_argument("--prefix", default="BPO-repro", help="wandb project prefix and state-file name")
    parser.add_argument("--only", default=None, help="regex on job names")
    parser.add_argument("--cpu", type=int, default=118, help="concurrent CPU jobs (128 physical cores)")
    parser.add_argument("--atari-per-gpu", type=int, default=12, help="concurrent Atari (and GPU SAC) jobs per GPU")
    parser.add_argument("--isaac-per-gpu", type=int, default=1,
                        help="concurrent IsaacLab jobs per GPU (3 per GPU measured slower in total than 1)")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--sac-device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--atari-device", default="cpu", choices=["cpu", "cuda"],
                        help="cpu keeps Atari off the GPUs; sharing slowed IsaacLab about 11x")
    parser.add_argument("--max-hours", default="", help="wall-clock caps per algorithm label, e.g. sac=24,dqn=24")
    parser.add_argument("--max-steps", type=int, default=None, help="cap SB3 total_timesteps (calibration)")
    parser.add_argument("--max-iters", type=int, default=None, help="cap IsaacLab iterations (calibration)")
    parser.add_argument("--stagger", type=float, default=2.0, help="seconds between launches")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    max_hours = {k: float(v) for k, v in (kv.split("=") for kv in args.max_hours.split(",") if kv)}
    jobs = build_jobs(args.sac_device, args.atari_device, max_hours)
    if args.only:
        jobs = [j for j in jobs if re.search(args.only, j.name)]
    assert len({j.name for j in jobs}) == len(jobs), "duplicate job names"

    if args.action == "list":
        groups = {}
        for j in jobs:
            groups.setdefault((j.pool, j.group), []).append(j)
        for (pool, group), js in groups.items():
            print(f"{pool:6s} {group:42s} {len(js):3d} runs")
        print(f"total: {len(jobs)} runs")
        return
    run(jobs, args)


if __name__ == "__main__":
    main()
