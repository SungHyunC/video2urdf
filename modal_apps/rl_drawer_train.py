"""Train the drawer-opening PPO policy (scripts/train_drawer_policy.py) on Modal CPUs.

CPU only: the policy is a small MLP and the cost is MuJoCo stepping, one environment per core.
The image carries only what training reads: video2urdf/, the two robot scripts it imports, the
trainer, the measured drawer (model.xml, design.json, meshes/) and Menagerie's Panda
(Apache-2.0). The policy, its observation scaling and the test results come back to
data/policies/<name>/ on this machine.

Run (ephemeral app, nothing deployed):
    .venv/bin/modal run modal_apps/rl_drawer_train.py::main --steps 5000000 --envs 32 --name drawer_ppo_honest
Then the trained policy on the full Panda arm from 20 unseen start poses, in parallel (no video):
    .venv/bin/modal run modal_apps/rl_drawer_train.py::arm --name drawer_ppo_honest --seeds 20
"""

import json
import time
from pathlib import Path

import modal

APP = "video2urdf-rl-drawer-train"
ROOT = Path(__file__).resolve().parent.parent
RUN = "data/tracks_design/drawer2_lidar"
PANDA = "third_party/mujoco_menagerie/franka_emika_panda"
REMOTE = "/root/v2u"


def _image() -> modal.Image:
    run = (ROOT / RUN).resolve()
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install("torch==2.15.0", index_url="https://download.pytorch.org/whl/cpu")
        .pip_install("mujoco==3.17.0", "stable-baselines3==2.9.2", "gymnasium==1.4.0", "numpy==2.5.4",
                     "scipy==1.18.1", "opencv-python-headless==5.0.0.93")
        .env({"MUJOCO_GL": "disable", "OMP_NUM_THREADS": "1"})
        .add_local_dir(ROOT / "video2urdf", f"{REMOTE}/video2urdf", ignore=["__pycache__", "*.pyc"])
        .add_local_file(ROOT / "scripts" / "robot_tasks.py", f"{REMOTE}/scripts/robot_tasks.py")
        .add_local_file(ROOT / "scripts" / "robot_demo.py", f"{REMOTE}/scripts/robot_demo.py")
        .add_local_file(ROOT / "scripts" / "train_drawer_policy.py", f"{REMOTE}/scripts/train_drawer_policy.py")
        .add_local_file(ROOT / "scripts" / "run_drawer_policy.py", f"{REMOTE}/scripts/run_drawer_policy.py")
        .add_local_file(run / "model.xml", f"{REMOTE}/{RUN}/model.xml")
        .add_local_file(run / "design.json", f"{REMOTE}/{RUN}/design.json")
        .add_local_dir(run / "meshes", f"{REMOTE}/{RUN}/meshes")
        .add_local_dir((ROOT / PANDA).resolve(), f"{REMOTE}/{PANDA}", ignore=["*.png", "video2urdf_*"])
    )
    return image


app = modal.App(APP, image=_image())


@app.function(cpu=32, memory=24576, timeout=4 * 3600)
def train(args: list[str]) -> dict:
    """Runs the trainer; returns its output files (name -> bytes) and the log."""
    import subprocess
    import sys

    out = Path("/tmp/policy")
    cmd = [sys.executable, f"{REMOTE}/scripts/train_drawer_policy.py", f"{REMOTE}/{RUN}", "--out", str(out), *args]
    print(" ".join(cmd), flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=REMOTE, capture_output=False, text=True)
    files = {p.name: p.read_bytes() for p in out.glob("*") if p.is_file()} if out.exists() else {}
    return {"returncode": proc.returncode, "seconds": time.time() - t0, "files": files}


@app.local_entrypoint()
def main(steps: int = 3_000_000, envs: int = 32, name: str = "drawer_ppo_honest", n_steps: int = 256,
         batch: int = 2048, epochs: int = 10, lr: float = 3e-4, ent: float = 0.0, seed: int = 0, test: int = 100):
    args = ["--steps", str(steps), "--envs", str(envs), "--n-steps", str(n_steps), "--batch", str(batch),
            "--epochs", str(epochs), "--lr", str(lr), "--ent", str(ent), "--seed", str(seed), "--test", str(test)]
    res = train.remote(args)
    dest = ROOT / "data" / "policies" / name
    dest.mkdir(parents=True, exist_ok=True)
    for fname, blob in res["files"].items():
        (dest / fname).write_bytes(blob)
    print(f"returncode {res['returncode']}, {res['seconds'] / 60:.1f} min on Modal; files -> {dest}: {sorted(res['files'])}")
    if "result.json" in res["files"]:
        r = json.loads(res["files"]["result.json"])
        for k in ("test_policy", "test_random"):
            print(k, json.dumps({kk: vv for kk, vv in r[k].items() if kk != "per_episode"}, indent=1))


@app.function(cpu=2, memory=4096, timeout=3600)
def retest(files: dict, test: int) -> dict:
    """The held-out test again on a saved policy (train_drawer_policy.py --eval-only)."""
    import subprocess
    import sys

    out = Path("/tmp/policy")
    out.mkdir(parents=True, exist_ok=True)
    for name, blob in files.items():
        (out / name).write_bytes(blob)
    cmd = [sys.executable, f"{REMOTE}/scripts/train_drawer_policy.py", f"{REMOTE}/{RUN}", "--out", str(out),
           "--eval-only", "--test", str(test)]
    proc = subprocess.run(cmd, cwd=REMOTE, capture_output=True, text=True)
    return {"returncode": proc.returncode, "stderr": proc.stderr[-3000:], "result": (out / "result.json").read_bytes()}


@app.local_entrypoint()
def reeval(name: str = "drawer_ppo_honest", test: int = 100):
    """Re-test data/policies/<name>/ on the held-out seeds with the current env bookkeeping."""
    src = ROOT / "data" / "policies" / name
    files = {n: (src / n).read_bytes() for n in ("policy.zip", "vecnormalize.pkl", "result.json") if (src / n).exists()}
    res = retest.remote(files, test)
    if res["returncode"] != 0:
        print(res["stderr"])
        return
    (src / "result.json").write_bytes(res["result"])
    r = json.loads(res["result"])
    for k in ("test_policy", "test_random"):
        print(k, json.dumps({kk: vv for kk, vv in r[k].items() if kk != "per_episode"}, indent=1))


@app.function(cpu=2, memory=4096, timeout=3600)
def arm_episode(policy_zip: bytes, vecnorm: bytes, seed: int, gain: float = 1.0) -> dict:
    """One episode of the policy on the full Panda arm (scripts/run_drawer_policy.py, no video)."""
    import subprocess
    import sys
    import tempfile

    tmp = Path(tempfile.mkdtemp())
    (tmp / "policy.zip").write_bytes(policy_zip)
    (tmp / "vecnormalize.pkl").write_bytes(vecnorm)
    cmd = [sys.executable, f"{REMOTE}/scripts/run_drawer_policy.py", f"{REMOTE}/{RUN}", "--policy", str(tmp / "policy.zip"),
           "--seed", str(seed), "--no-video", "--result", str(tmp / "arm.json"), "--arm-gain", str(gain)]
    proc = subprocess.run(cmd, cwd=REMOTE, capture_output=True, text=True)
    if proc.returncode != 0:
        return {"seed": seed, "error": proc.stderr[-2000:]}
    return json.loads((tmp / "arm.json").read_text())


@app.local_entrypoint()
def arm(name: str = "drawer_ppo_honest", seeds: int = 20, seed0: int = 20_000, gain: float = 1.0, tag: str = ""):
    """The policy on the full arm from `seeds` start poses it never saw, in parallel; writes
    data/policies/<name>/arm_eval.json."""
    src = ROOT / "data" / "policies" / name
    blob, norm = (src / "policy.zip").read_bytes(), (src / "vecnormalize.pkl").read_bytes()
    runs = list(arm_episode.map([blob] * seeds, [norm] * seeds, range(seed0, seed0 + seeds), [gain] * seeds))
    ok = [r for r in runs if "error" not in r]
    travel = sum(r["travel_mm"] for r in ok)
    summary = {"episodes": len(runs), "errors": len(runs) - len(ok),
               "success_rate": sum(r["success"] for r in ok) / max(len(ok), 1),
               "drawer_mm_held_median": sorted(r["drawer_mm_held"] for r in ok)[len(ok) // 2] if ok else None,
               "held_share_of_travel": sum(r["travel_mm"] * (r["held_share_of_travel"] or 0) for r in ok) / travel
               if travel else None,
               "finger_open_on_hold_min_mm": min((r["finger_open_on_hold_min_mm"] for r in ok
                                                  if r["finger_open_on_hold_min_mm"] is not None), default=None),
               "deepest_contact_not_handle_mm_max": max((r["deepest_contact_not_handle_mm"] for r in ok), default=None),
               "net_force_on_drawer_max_n": max((r["net_force_on_drawer_max_n"] for r in ok), default=None),
               "net_force_on_cabinet_max_n": max((r["net_force_on_cabinet_max_n"] for r in ok), default=None),
               "net_force_on_drawer_step_avg_max_n": max((r["net_force_on_drawer_step_avg_max_n"] for r in ok), default=None),
               "net_force_on_drawer_step_avg_median_n": sorted(r["net_force_on_drawer_step_avg_max_n"] for r in ok)[len(ok) // 2]
               if ok else None,
               "steps_with_step_avg_force_over_60n_total": sum(r["steps_with_step_avg_force_over_60n"] for r in ok),
               "ik_error_max_mm": max((r["ik_error_max_mm"] or 0 for r in ok), default=None),
               "hand_tilt_max_deg": max((r["hand_tilt_max_deg"] for r in ok), default=None),
               "seeds": f"{seed0}..{seed0 + seeds - 1} (hand start poses, drawn like the env's)", "arm_gain": gain}
    (src / f"arm_eval{tag}.json").write_text(json.dumps({"summary": summary, "episodes": runs}, indent=1))
    print(json.dumps(summary, indent=1))
