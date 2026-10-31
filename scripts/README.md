# scripts/

Command-line entry points, grouped by what you'd use them for. Each script's docstring has its
usage. Run them with the project's environment: `.venv/bin/python scripts/<name>.py ...`.

## Build a model from a video
| Script | What it does |
|---|---|
| `run_agent.py` | The Muse Spark agent: a video or an iPhone LiDAR / RGB-D capture in, a measured URDF + MJCF out (then Spark's shape design). `--open-weights`, `--glimmer` |
| `video_to_model.py` | One command for an RGB-D capture or a plain phone video (MapAnything depth + automatic scale), through to a viewer folder |
| `video_to_capture.py` | A plain phone video -> an RGB-D capture in the Stray Scanner layout (MapAnything on Modal) |
| `calibrate_capture.py` | A calibrated copy of a MapAnything capture from one known length |

## Improve a fitted model
| Script | What it does |
|---|---|
| `design_shape.py` | Spark names the object and rebuilds each link from clean primitives, checked over the video |
| `add_physics.py` | Masses, inertia and joint friction (Spark judges materials; friction from still spans) |
| `texture_model.py` | Paints each face with the video frame that saw it best |
| `sam3d_meshes.py` | Realistic link meshes from SAM 3D Objects, checked against the LiDAR depth |
| `infer_appearance.py`, `resume_appearance.py`, `transfer_sam3d_texture.py` | Appearance completion for hidden faces (experimental) |

## Use the model
| Script | What it does |
|---|---|
| `robot_tasks.py` | A Franka Panda grasps and works the measured object in MuJoCo (close / open, pull a drawer, swing an arm, carry) |
| `train_drawer_policy.py`, `run_drawer_policy.py` | PPO policy on the measured drawer (trained on Modal: `modal_apps/rl_drawer_train.py`) and its run on the full arm |
| `urdf_doctor.py` | Checks an existing URDF's joint against the motion measured from video |
| `physics_demo.py`, `turntable.py` | Videos: the measured friction holding a joint; a camera circling the model |
| `robot_demo.py` | The earlier push-only robot demo (kept for the old videos) |

## Benchmarks and checks
| Script | What it does |
|---|---|
| `rbo_download.py`, `rbo_bench.py`, `rbo_eval.py` | The RBO motion-capture benchmark: download, run the agent, score exactly the agent's fit |
| `refit_lidar.py` | Re-run the agent's LiDAR fit on saved tracks, in a copy (regression checks) |
| `make_synth.py`, `eval_tracks.py`, `fit3d.py` | Synthetic videos with known joints, and scoring against them |
| `eval_pseudo_depth.py`, `compare_hinges.py` | MapAnything depth and hinge fits against the LiDAR capture |
| `check_sam_open.py`, `check_fast_mask.py`, `smoke_test.py` | Backend checks: open-weights SAM, the mask decoder, one cheap call per API |

## Viewer and demo
| Script | What it does |
|---|---|
| `export_viewer.py` | Exports runs to the web viewer in `docs/` (GitHub Pages) |
| `make_demo.py` | Cuts the demo video from the project's outputs (`docs/demo.mp4`, `docs/demo_script.md`) |
| `make_reel.py`, `make_viewer.py` | A single run's reel; a local review page |

## Lower-level steps (used by the agent; handy for debugging)
| Script | What it does |
|---|---|
| `track_video.py` | SAM 3.1 tracking of named parts, with an overlay video |
| `fit_model.py` | 2D tracks -> joints -> URDF/MJCF -> replay |
| `fit_lidar_hinge.py`, `build_lidar_model.py` | The LiDAR hinge fit and model build, step by step |
| `experimental_drawer_collision.py`, `experimental_laptop_align.py`, `render_experimental_replay.py` | Review-only experiments on copies of a run |
| `fetch_assets.sh` | Downloads the Franka Panda (MuJoCo Menagerie) into `third_party/` |
