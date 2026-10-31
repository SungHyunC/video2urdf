"""Train the drawer-opening PPO policy (scripts/train_drawer_policy.py) on Modal CPUs.

CPU only: the policy is a small MLP and the cost is MuJoCo stepping, one environment per core.
The image carries only what training reads: video2urdf/, the two robot scripts it imports, the
trainer, the measured drawer (model.xml, design.json, meshes/) and Menagerie's Panda
(Apache-2.0). The policy, its observation scaling and the test results come back to
data/policies/<name>/ on this machine.

Run (ephemeral app, nothing deployed):
    .venv/bin/modal run modal_apps/rl_drawer_train.py --steps 3000000 --envs 32 --name drawer_ppo_honest
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
