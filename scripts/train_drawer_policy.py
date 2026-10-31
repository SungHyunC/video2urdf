"""Train a drawer-opening policy with PPO on a drawer measured from video, then test it.

The environment (video2urdf/rl_drawer.py) is the fitted cabinet and drawer with a Panda hand;
every episode moves and turns the cabinet and rescales the drawer's measured friction. After
training, the policy runs 100 test episodes on placements it never saw (fixed seeds), next to
a random policy for scale.

Usage: .venv/bin/python scripts/train_drawer_policy.py data/tracks_design/drawer2_lidar
           [--steps 1500000] [--out data/policies/drawer_ppo]
Needs the rl extra: uv sync --extra rl
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def make(run: str, seed: int):
    def thunk():
        from stable_baselines3.common.monitor import Monitor
        from video2urdf.rl_drawer import DrawerEnv
        return Monitor(DrawerEnv(run, seed=seed), info_keywords=("success", "drawer_m"))
    return thunk


def evaluate(policy, run: str, episodes: int, seed0: int = 10_000, norm=None) -> dict:
    """Success rate and drawer travel on held-out placements; policy=None acts at random.
    ``norm``: the VecNormalize the policy was trained with (observations are scaled by it)."""
    from video2urdf.rl_drawer import DrawerEnv
    env = DrawerEnv(run, seed=seed0)
    ok, travel, steps = [], [], []
    for k in range(episodes):
        obs, _ = env.reset(seed=seed0 + k)
        for t in range(10_000):
            o = norm.normalize_obs(obs) if norm is not None else obs
            a = env.action_space.sample() if policy is None else policy.predict(o, deterministic=True)[0]
            obs, r, term, trunc, info = env.step(a)
            if term or trunc:
                break
        ok.append(info["success"])
        travel.append(info["drawer_m"])
        steps.append(t + 1)
    return {"episodes": episodes, "success_rate": float(np.mean(ok)), "drawer_mm_mean": float(1000 * np.mean(travel)),
            "steps_to_success_median": float(np.median([s for s, o in zip(steps, ok) if o])) if any(ok) else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--steps", type=int, default=2_000_000)
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "policies" / "drawer_ppo")
    args = ap.parse_args()
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

    args.out.mkdir(parents=True, exist_ok=True)
    # observations span metres and millimetres (the hand's position, the handle 18 mm tall): normalise them
    venv = VecNormalize(SubprocVecEnv([make(str(args.run), seed=i) for i in range(args.envs)]),
                        norm_obs=True, norm_reward=True, clip_obs=10.0)
    curve = []

    class Log(BaseCallback):
        """Training success rate over the last 200 finished episodes, every ~50k steps."""
        def __init__(self):
            super().__init__()
            self.recent, self.next = [], 50_000

        def _on_step(self):
            for info in self.locals["infos"]:
                if "episode" in info:
                    self.recent = (self.recent + [info.get("success", False)])[-200:]
            if self.num_timesteps >= self.next and self.recent:
                curve.append({"steps": self.num_timesteps, "success_rate": float(np.mean(self.recent))})
                print(f"{self.num_timesteps:>9} steps: training success {np.mean(self.recent):.2f}", flush=True)
                self.next += 50_000
            return True

    model = PPO("MlpPolicy", venv, n_steps=1024, batch_size=2048, learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
                n_epochs=10, ent_coef=0.0, policy_kwargs={"net_arch": [256, 256]}, seed=0, verbose=0)
    t0 = time.time()
    model.learn(total_timesteps=args.steps, callback=Log())
    minutes = (time.time() - t0) / 60
    model.save(args.out / "policy")
    venv.save(str(args.out / "vecnormalize.pkl"))
    venv.training = False
    result = {"run": str(args.run), "steps": args.steps, "train_minutes": round(minutes, 1), "curve": curve,
              "test_policy": evaluate(model, str(args.run), 100, norm=venv),
              "test_random": evaluate(None, str(args.run), 20),
              "randomized": "cabinet +-3 cm, +-10 deg; drawer friction x0.5-1.5; hand start within 10 x 20 x 15 cm",
              "success": "drawer pulled out at least 150 mm (of 248 mm measured)"}
    venv.close()
    (args.out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "curve"}, indent=2))


if __name__ == "__main__":
    main()
