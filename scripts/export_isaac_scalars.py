"""Export the TensorBoard scalars of the IsaacLab variant runs to CSV, so the data can be read
without the (gitignored) logs/ directory.

Reads   logs/rsl_rl/<experiment>/<time>_<env>-<variant>-s<seed>/events*
Writes  abpo_results/isaac_scalars/<env>/<variant>/s<seed>.csv  (one row per iteration, one column per tag)

If a run name was used twice (a restarted run), the latest directory wins.

    python scripts/export_isaac_scalars.py
"""
import glob
import os
import re

import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = "logs/rsl_rl"
OUT = "abpo_results/isaac_scalars"


def main():
    runs = {}
    for run_dir in sorted(glob.glob(os.path.join(ROOT, "*", "*"))):  # sorted: later time stamps overwrite
        m = re.match(r"[\d_-]+_(\w+_rough)-(.+)-s(\d+)$", os.path.basename(run_dir))
        if m and glob.glob(os.path.join(run_dir, "events*")):
            runs[(m[1], m[2], int(m[3]))] = run_dir

    for (env, variant, seed), run_dir in sorted(runs.items()):
        ea = EventAccumulator(run_dir, size_guidance={"scalars": 0})
        ea.Reload()
        # "<tag>/time" copies are indexed by wall-clock seconds, not iteration: skip them
        tags = [t for t in ea.Tags()["scalars"] if not t.endswith("/time")]
        cols = {tag: pd.Series({e.step: e.value for e in ea.Scalars(tag)}) for tag in tags}
        if not cols:
            continue
        df = pd.DataFrame(cols).rename_axis("iteration")
        path = os.path.join(OUT, env, variant, f"s{seed}.csv")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, float_format="%.6g")
        print(f"{path}: {len(df)} iterations, {df.shape[1]} metrics  <- {run_dir}")


if __name__ == "__main__":
    main()
