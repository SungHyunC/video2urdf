# How it works

The pipeline, the two capture paths, which Meta model decides what, and where the code is.
Results are in [RESULTS.md](RESULTS.md), commands in [COMMANDS.md](COMMANDS.md). The agent's
tools and its finish gate are in the [README](../README.md#the-agent).

## The pipeline

```
phone video (or iPhone LiDAR capture)
  -> Muse Spark watches the video, names the parts, prompts SAM 3.1   (agent, tool calls)
  -> SAM 3.1 finds each part on key frames and tracks it through the video
  -> joint fit: hinge or slide, axis, position, limits, from the parts' relative motion
  -> link geometry: each part's silhouette (2D) or depth (LiDAR), painted with the video
  -> URDF + MJCF
  -> MuJoCo replay rendered from the camera, scored against the video (coverage per frame);
     the fit tool names a wrong part choice (a base that holds the lid, a lid track that
     slid onto the closed lid's back)
  -> Muse Spark watches the replay, re-segments / re-tracks / refits
  -> physics: Spark judges materials and masses; joint friction measured where a part
     stands still with no hand on it, checked by letting the model go in MuJoCo
  -> report with the measured sizes and what is uncertain
```

The model is then used: a Franka Panda in MuJoCo grasps the measured object and works it (closes
and reopens a lid, pulls a drawer out by its handle and pushes it back, swings a lamp's arm,
carries a mouse; `scripts/robot_tasks.py`).

## Two capture paths

| | 2D (any phone) | 3D (iPhone LiDAR via Stray Scanner, or any RGB-D camera) |
|---|---|---|
| View | tripod, camera roughly facing the motion plane | any angle, handheld is fine |
| Scale | an ID-1 card (85.6 x 54 mm) in view; use a loyalty card, never a payment card | metric depth, no card |
| Joint fit | planar: least-squares hinge centre / slide direction from relative poses | 3D: plate planes per frame, hinge = where plate and base planes cross; slides by ICP |
| Geometry | voted silhouettes extruded to meshes | sizes from depth (lid outline, base thickness, ...) |

A plain video with a moving or oblique camera can take the 3D path too: MapAnything estimates
its depth and camera poses (experimental; see [RESULTS.md](RESULTS.md#plain-video-without-lidar-mapanything)).
`scripts/video2urdf.py` picks the path from the capture.

## Which Meta model decides what

| Model | Where | Role |
|---|---|---|
| **Muse Spark** (`muse-spark-1.3-contributor`, Meta Model API) | `video2urdf/agent.py`, `scripts/add_physics.py` | The agent: watches the whole video (mp4 input), looks at frames, chooses noun prompts for SAM, picks the parts' roles, calls track / fit / fit_lidar, watches the replay (the model drawn over the video) to judge the fit over time, refits, then judges each link's material and mass and finds when a part stood still with no hand on it (`set_physics`), and writes the report. After the fit it names the object and designs its clean visual from primitives, checking each design over the video (`video2urdf/designer.py`). Chat Completions with tool calls, images, video and structured output. |
| **SAM 3.1** (`sam-3.1`, Meta Model API) | `video2urdf/sam.py`, `scripts/track_video.py` | Concept prompts on key frames find each part; one box-anchored video request tracks all parts with stable IDs. |
| **Muse Spark Open** (open weights, Baseten) | `run_agent.py --open-weights` | The same agent on open weights (it takes video too). |
| **Muse Glimmer 30B** (open weights `meta-models/Muse-Glimmer-30B`, our own Modal Dedicated Endpoint, SGLang) | `run_agent.py --glimmer` | The same agent on a 30B dense open model we serve ourselves: tool calls and images (the endpoint drops video, so it works from frames). |
| **SAM 3.1 / SAM 3** (open weights `facebook/sam3.1`, `facebook/sam3`, on a Modal GPU) | `modal_apps/sam3_open.py`, `video2urdf/sam_open.py` | Open-weights segmentation (SAM 3.1 concept prompts) and box tracking (SAM 3's tracker), answering in the hosted API's wire format so the rest of the pipeline is unchanged. |
| **SAM 3D Objects** (open weights `facebook/sam-3d-objects`, on a Modal GPU) | `modal_apps/sam3d.py`, `scripts/sam3d_meshes.py` | A realistic, coloured mesh of each link from one frame and its SAM 3.1 mask, laid out in metres by our LiDAR depth (given as its pointmap), checked against the measured depth points; it becomes the link's visual, the measured shape stays its collision. |
| **MapAnything** (open weights `facebook/map-anything-apache`, on a Modal GPU) | `modal_apps/mapanything_open.py`, `video2urdf/pseudo_depth.py` | Metric depth, intrinsics and camera poses from a plain RGB video, written as a LiDAR-style capture so the 3D path runs without LiDAR (experimental, see [RESULTS.md](RESULTS.md#plain-video-without-lidar-mapanything)). |

Every model called at runtime is a Meta model. Switching between the Meta API and open
weights is one flag.

## Tracking with SAM 3.1

- A concept prompt on a whole video starts tracking only where the concept is first confident
  and then goes forward, so parts get missed early. Instead the agent finds each part with a
  noun prompt on a few key frames (image calls), then one box-anchored video request tracks
  every part with stable IDs (`video2urdf/sam.py`).
- The API's `propagation_direction=both` dropped the backward pass without an error, so each
  track runs forward, and forward again over a reversed clip of the frames before the first
  anchor; the two passes run at once and are merged per frame and object.
- A part can be anchored on every key frame where it was found, so a part that leaves the frame
  or hides behind a hand is picked up again. A mask that shrinks to a few pixels counts as lost.
- Videos above 40 fps are tracked on every second frame. Numbers:
  [RESULTS.md](RESULTS.md#faster-tracking-and-scale-without-a-reference).

## Code map

`video2urdf/` is the library; `scripts/` holds the command-line entry points
([scripts/README.md](../scripts/README.md)); `modal_apps/` the GPU apps on Modal.

| Stage | Modules |
|---|---|
| One command: input, checks, routing, summary | `run.py` (the whole run), `ingest.py` (what the input is, normalised from its content), `capture_check.py` (reshoot tips before anything is paid for), `route.py` (which path), `fallback.py` (a rigid model when no joint is found), `summary.py` (what was measured and what was estimated) |
| The agent | `agent.py` (Muse Spark's tools, turn budget and finish gate) |
| Segmentation and tracking | `sam.py` (SAM 3.1 on the Meta Model API), `sam_open.py` (the same on open weights, `modal_apps/sam3_open.py`), `fast_mask.py` (compiled mask decoder), `track_checks.py` |
| Planar fit (2D video) | `motion.py` (per-part 2D motion from masks), `joints.py` (joints and kinematic tree), `shape.py` (link profiles extruded to meshes), `emit.py` (URDF/MJCF), `pipeline.py` (tracks to replay in one call) |
| Depth fits (LiDAR, RGB-D, MapAnything) | `stray.py` (Stray Scanner captures), `lidar.py` (hinged plates, doors, slides, drawers), `chain3d.py` (rods on hinges), `static_shape.py` (rigid objects), `lift3d.py` (3D joints from video plus depth), `emit3d.py` (3D URDF/MJCF), `pseudo_depth.py` and `autoscale.py` (MapAnything depth and its scale), `calibration.py` (scale from a known length) |
| Checking the model against the video | `replay.py` (MuJoCo replay over the video, scored per frame) |
| Making the model usable | `physics.py` (mass, inertia, joint friction), `texture.py` (painted with the video), `design.py` and `designer.py` (Spark's shape design), `completion.py` (the hidden body of a partly seen object), `portable.py` and `urdf_checks.py` (downloads that load in other simulators) |
| Evaluation and uses | `synth.py` (synthetic videos with known joints), `rbo.py` (the RBO benchmark), `doctor.py` (URDF Doctor), `spark_alone.py` (the no-tools baseline), `rl_drawer.py` (the drawer RL task) |
| Experimental appearance completion | `appearance.py`, `appearance_check.py`, `assessment.py`, `planar_check.py`, `planar_shape.py`, `texture_transfer.py` ([log](appearance_validation.md)) |

## How this differs from research on articulated-object models

- Ditto (CVPR 2022) and PARIS (ICCV 2023) reconstruct an object's parts and joint from
  observations of two static states.
- URDFormer (RSS 2024) predicts a URDF's structure from a single image and fills it with template
  parts. Articulate-Anything (2024) has a vision-language model write the articulation from text,
  images or video, with part meshes retrieved from a dataset.
- Video2URDF takes one casual video of the real object being worked. It measures each joint's
  axis, position and limits in metres from the parts' tracked motion, and replays the model over
  the same video to check it. It says which numbers were measured and which were estimated. URDF
  Doctor runs the same measurement against a URDF you already have.
