"""Train a drawer-opening policy with PPO on a drawer measured from video, then test it.

The environment (video2urdf/rl_drawer.py) is the fitted cabinet and drawer, with their measured
collision on, and a Panda hand on a force-limited mount; every episode moves and turns the
cabinet and rescales the drawer's measured friction. After training, the policy runs 100 test
episodes on placements it never saw (fixed seeds), next to a random policy on the same seeds,
and the test reports what keeps it honest: the share of the drawer's travel made while the
fingers pinched the handle, how far the fingers stayed open on the hold, the hand's deepest
contact and largest push on anything but the handle, and whether the drawer left its range.

Usage: .venv/bin/python scripts/train_drawer_policy.py data/tracks_design/drawer2_lidar
           [--steps 3000000] [--envs 8] [--out data/policies/drawer_ppo_honest]
Needs the rl extra: uv sync --extra rl. On Modal: modal_apps/rl_drawer_train.py.
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
        return Monitor(DrawerEnv(run, seed=seed), info_keywords=("success", "drawer_m", "stats"))
    return thunk


def evaluate(policy, run: str, episodes: int, seed0: int = 10_000, norm=None) -> dict:
    """Success and honesty on held-out placements; policy=None acts at random.
    ``norm``: the VecNormalize the policy was trained with (observations are scaled by it)."""
    from video2urdf.rl_drawer import DrawerEnv
    env = DrawerEnv(run, seed=seed0)
    rng = np.random.default_rng(seed0)
    eps = []
    for k in range(episodes):
        obs, _ = env.reset(seed=seed0 + k)
        first_success = None
        for t in range(10_000):
            o = norm.normalize_obs(obs) if norm is not None else obs
            a = rng.uniform(-1, 1, 4) if policy is None else policy.predict(o, deterministic=True)[0]
            obs, r, term, trunc, info = env.step(a)
            if first_success is None and info["success"]:
                first_success = t + 1
            if term or trunc:
                break
        s = info["stats"]
        eps.append({"seed": seed0 + k, "success": bool(info["success"]), "drawer_mm": 1000 * info["drawer_m"],
                    "drawer_end_mm": 1000 * info["drawer_now_m"], "steps_to_success": first_success,
                    "travel_mm": 1000 * s["travel_m"], "held_travel_mm": 1000 * s["held_travel_m"],
                    "held_steps": s["held_steps"],
                    "finger_open_held_min_mm": None if s["finger_open_held_min_m"] is None else 1000 * s["finger_open_held_min_m"],
                    "finger_open_min_mm": 1000 * s["finger_open_min_m"], "deepest_other_mm": 1000 * s["deepest_other_m"],
                    "deepest_other_what": s["deepest_other_what"], "push_force_max_n": s["push_force_max_n"],
                    "drawer_min_mm": 1000 * s["q_min_m"], "drawer_peak_mm": 1000 * s["q_peak_m"]})
    return summarize(eps)


def summarize(eps: list[dict]) -> dict:
    ok = [e for e in eps if e["success"]]
    travel = sum(e["travel_mm"] for e in eps)
    held = sum(e["held_travel_mm"] for e in eps)
    grips = [e["finger_open_held_min_mm"] for e in eps if e["finger_open_held_min_mm"] is not None]
    deep = [e["deepest_other_mm"] for e in eps]
    push = [e["push_force_max_n"] for e in eps]
    worst = max(eps, key=lambda e: e["deepest_other_mm"])
    return {
        "episodes": len(eps), "success_rate": len(ok) / len(eps),
        "drawer_mm_mean": float(np.mean([e["drawer_mm"] for e in eps])),
        "drawer_mm_successful_median": float(np.median([e["drawer_mm"] for e in ok])) if ok else None,
        "steps_to_success_median": float(np.median([e["steps_to_success"] for e in ok])) if ok else None,
        "honesty": {
            "travel_mm_total": travel, "held_travel_mm_total": held,
            "held_share_of_travel": held / travel if travel > 0 else None,
            "held_share_min_over_moving_episodes": min((e["held_travel_mm"] / e["travel_mm"] for e in eps
                                                        if e["travel_mm"] > 5), default=None),
            "finger_open_on_hold_min_mm": min(grips) if grips else None,
            "finger_open_on_hold_median_of_episode_min_mm": float(np.median(grips)) if grips else None,
            "finger_open_min_any_mm": min(e["finger_open_min_mm"] for e in eps),
            "deepest_contact_not_handle_mm_max": max(deep), "deepest_contact_not_handle_mm_median": float(np.median(deep)),
            "deepest_contact_not_handle_what": worst["deepest_other_what"], "deepest_contact_seed": worst["seed"],
            "push_force_not_handle_n_max": max(push), "push_force_not_handle_n_median": float(np.median(push)),
            "drawer_min_mm": min(e["drawer_min_mm"] for e in eps), "drawer_peak_mm": max(e["drawer_peak_mm"] for e in eps),
        },
        "per_episode": eps,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--steps", type=int, default=3_000_000)
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--n-steps", type=int, default=1024, help="rollout length per env")
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ent", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test", type=int, default=100, help="held-out test episodes")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "policies" / "drawer_ppo_honest")
    args = ap.parse_args()
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

    torch.set_num_threads(min(8, torch.get_num_threads()))
    args.out.mkdir(parents=True, exist_ok=True)
    # observations span metres and millimetres (the hand's position, the handle 18 mm tall): normalise them
    venv = VecNormalize(SubprocVecEnv([make(str(args.run), seed=args.seed * 1000 + i) for i in range(args.envs)]),
                        norm_obs=True, norm_reward=True, clip_obs=10.0)
    curve = []

    class Log(BaseCallback):
        """Training success, drawer travel and held share over the last 200 finished episodes, every ~50k steps."""
        def __init__(self):
            super().__init__()
            self.recent, self.next, self.t0 = [], 50_000, time.time()

        def _on_step(self):
            for info in self.locals["infos"]:
                if "episode" in info:
                    s = info.get("stats") or {}
                    self.recent = (self.recent + [(info.get("success", False), info.get("drawer_m", 0.0),
                                                   s.get("travel_m", 0.0), s.get("held_travel_m", 0.0),
                                                   info["episode"]["r"])])[-200:]
            if self.num_timesteps >= self.next and self.recent:
                r = np.array(self.recent, float)
                travel = r[:, 2].sum()
                row = {"steps": self.num_timesteps, "minutes": round((time.time() - self.t0) / 60, 1),
                       "success_rate": float(r[:, 0].mean()), "drawer_mm_mean": float(1000 * r[:, 1].mean()),
                       "held_share": float(r[:, 3].sum() / travel) if travel > 1e-6 else None,
                       "return_mean": float(r[:, 4].mean())}
                curve.append(row)
                print(json.dumps(row), flush=True)
                self.next += 50_000
            return True

    model = PPO("MlpPolicy", venv, n_steps=args.n_steps, batch_size=args.batch, learning_rate=args.lr, gamma=0.99,
                gae_lambda=0.95, n_epochs=args.epochs, ent_coef=args.ent, policy_kwargs={"net_arch": [256, 256]},
                seed=args.seed, verbose=0, device="cpu")
    t0 = time.time()
    model.learn(total_timesteps=args.steps, callback=Log())
    minutes = (time.time() - t0) / 60
    model.save(args.out / "policy")
    venv.save(str(args.out / "vecnormalize.pkl"))
    venv.training = False
    venv.norm_reward = False
    t1 = time.time()
    test_policy = evaluate(model, str(args.run), args.test, norm=venv)
    test_random = evaluate(None, str(args.run), args.test)
    result = {"run": str(args.run), "steps": args.steps, "envs": args.envs, "train_minutes": round(minutes, 1),
              "test_minutes": round((time.time() - t1) / 60, 1),
              "ppo": {"n_steps": args.n_steps, "batch": args.batch, "epochs": args.epochs, "lr": args.lr,
                      "ent_coef": args.ent, "gamma": 0.99, "gae_lambda": 0.95, "net": [256, 256], "seed": args.seed},
              "curve": curve, "test_policy": test_policy, "test_random": test_random,
              "test_seeds": f"{10_000}..{10_000 + args.test - 1} (training envs draw from seeds {args.seed * 1000}.."
                            f"{args.seed * 1000 + args.envs - 1})",
              "randomized": "cabinet +-3 cm, +-10 deg; drawer friction x0.5-1.5; hand start within 10 x 20 x 15 cm",
              "success": "drawer pulled out at least 150 mm (of 248 mm measured) while the fingers pinch the handle",
              "env": "measured drawer + cabinet collision on; Panda hand on a force-limited Cartesian mount "
                     "(3000 N/m, <= 60 N per axis); stiff end stops and hand contacts"}
    venv.close()
    (args.out / "result.json").write_text(json.dumps(result, indent=2))
    brief = {k: v for k, v in result.items() if k not in ("curve", "test_policy", "test_random")}
    for name in ("test_policy", "test_random"):
        brief[name] = {k: v for k, v in result[name].items() if k != "per_episode"}
    print(json.dumps(brief, indent=2))


if __name__ == "__main__":
    main()
