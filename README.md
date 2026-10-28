# Video2URDF

**One short video of a real articulated object in, a measured URDF/MJCF out, and a MuJoCo
replay over the video that shows the model moves like the real thing.**

Robotics developers who need a simulation model of a real object (a laptop lid, a desk lamp,
a drawer, a box flap) usually model it by hand: guess the hinge, guess the limits, tweak until
it looks right. Video2URDF measures it instead. Film the object being opened and closed; a
Muse Spark agent finds the parts with SAM 3.1, tracks them, fits each joint by least squares,
writes URDF and MJCF, replays the model in MuJoCo from the camera's own viewpoint and checks
the overlay, frame by frame, before it reports.

Built solo for the Meta Global AI Developer Hackathon (Oct 26 - Nov 4 2026). All code in this
repository was written during the hackathon; see [Built on](#built-on) for what it uses.

**Live viewer:** https://sunghyunc.github.io/video2urdf/ (the phone video, the MuJoCo replay over it,
the robot demo, and the measured model in 3D with its URDF/MJCF).

## What it does

```
phone video (or iPhone LiDAR capture)
  -> Muse Spark looks at frames, names the parts, prompts SAM 3.1   (agent, tool calls)
  -> SAM 3.1 finds each part on key frames and tracks it through the video
  -> joint fit: hinge or slide, axis, position, limits, from the parts' relative motion
  -> link geometry: each part's silhouette (2D) or depth (LiDAR), with mass and inertia
  -> URDF + MJCF
  -> MuJoCo replay rendered from the camera, scored against the video (IoU per frame)
  -> Muse Spark reads the scores and the worst frames, re-segments / re-tracks / refits,
     and writes a report with the measured sizes and what is uncertain
```

The model is then used: a Franka Panda in MuJoCo closes the measured laptop by pushing along
the measured hinge (`scripts/robot_demo.py`).

Two capture paths:

| | 2D (any phone) | 3D (iPhone LiDAR via Stray Scanner, or any RGB-D camera) |
|---|---|---|
| View | tripod, camera roughly facing the motion plane | any angle, handheld is fine |
| Scale | an ID-1 card (85.6 x 54 mm) in view; use a loyalty card, never a payment card | metric depth, no card |
| Joint fit | planar: least-squares hinge centre / slide direction from relative poses | 3D: plate planes per frame, hinge = where plate and base planes cross; slides by ICP |
| Geometry | voted silhouettes extruded to meshes | sizes from depth (lid outline, base thickness, ...) |

## Meta models, and what each does

| Model | Where | Role |
|---|---|---|
| **Muse Spark** (`muse-spark-1.3-contributor`, Meta Model API) | `video2urdf/agent.py` | The agent: looks at frames, chooses noun prompts for SAM, picks the parts' roles, calls track / fit / fit_lidar, judges the replay images and numbers, refits, writes the report. Chat Completions with tool calls and images. |
| **SAM 3.1** (`sam-3.1`, Meta Model API) | `video2urdf/sam.py`, `scripts/track_video.py` | Concept prompts on key frames find each part; one box-anchored video request tracks all parts with stable IDs. |
| **Muse Spark Open** (open weights, Baseten) | `run_agent.py --open-weights` | The same agent on open weights. |
| **SAM 3.1 / SAM 3** (open weights `facebook/sam3.1`, `facebook/sam3`, on a Modal GPU) | `modal_apps/sam3_open.py`, `video2urdf/sam_open.py` | Open-weights segmentation (SAM 3.1 concept prompts) and box tracking (SAM 3's tracker), answering in the hosted API's wire format so the rest of the pipeline is unchanged. |

Every model called at runtime is a Meta model. Switching between the Meta API and open
weights is one flag.

## Results

### Synthetic (MuJoCo-rendered videos with known joints)

| Case | Joint type | Axis / position error | Angle error |
|---|---|---|---|
| Lamp, 2 hinges, 2D | correct tree and types | hinge position 1.7 / 2.5 mm | 0.08 / 0.38 deg mean |
| Drawer, 2D | prismatic | direction 0.00 deg, position 3.2 mm | |
| Lamp seen obliquely, handheld wobble, 3D | | shoulder 0.5-0.85 deg / 4 mm, elbow 1.4-1.6 deg / 8 mm | 0.1 / 0.9 deg |
| Drawer sliding toward the camera, 3D | prismatic | direction 0.24 deg | travel 0.17 mm |

### Real objects, filmed for this project

- **13" MacBook Pro, iPhone LiDAR, no card**, agent from scratch: lid 309 x 205 mm (spec
  304 x 212), base 14 mm, lid 3.8 mm, opening about 0..110 deg, hinge drawn on the real hinge;
  the replayed model covers 95% (lid) and 94% (base) of SAM's masks over the 923 frames, while
  the phone moves. The Panda then closes it to within 1 deg of its measured limit.
- Laptop, 2D side view: one hinge, 0-77 deg, fit residual 1.7 mm. Spark saw its first fit was
  wrong (base tracked on 319/408 frames, 25 deg range), re-segmented on other frames and
  re-tracked: 408/408 frames.
- Monitor stand, 2D: base -> lower arm (0-30 deg) -> upper arm (-81-0 deg), residuals
  1.0 / 0.9 mm.

### Benchmark: RBO dataset (real RGB-D with motion capture)

The [RBO dataset](https://zenodo.org/records/1036660) (TU Berlin, CC BY 4.0) has Kinect RGB-D
videos of people operating articulated objects, with motion-captured rigid bodies. The agent
runs each sequence from scratch (same prompts and tools as on our own videos); the result is
scored against the plate's motion relative to the base from motion capture. The operators
often push the whole object around while opening it (up to 15 cm and 47 deg), so the base is
followed too.

| Sequence | Axis direction | Axis position | Angle error, median | Angle error, p90 | Frames measured |
|---|---|---|---|---|---|
| laptop01 | 2.0 deg | 24 mm | 1.9 deg | 31.7 deg | 186/186 |
| laptop02 | 2.1 deg | 31 mm | 3.3 deg | 8.1 deg | 203/205 |
| laptop04 | 2.2 deg | 12 mm | 2.0 deg | 4.0 deg | 176/192 |
| laptop05 | 0.9 deg | 19 mm | 2.5 deg | 5.4 deg | 184/185 |
| book03 | 2.5 deg | 18 mm | 1.3 deg | 3.3 deg | 188/329 |
| book05 | 2.8 deg | 63 mm | 1.4 deg | 54.0 deg | 215/256 |
| book13 | 6.7 deg | 92 mm | 16.0 deg | 45.9 deg | 124/282 |
| treasurebox03 | 11.5 deg | 37 mm | 0.9 deg | 7.3 deg | 88/224 |
| cardboardbox03 | 13.3 deg | 151 mm | 5.5 deg | 34.0 deg | 130/265 |
| cardboardbox06 | 2.5 deg | 47 mm | 40.8 deg | 132.9 deg | 187/222 |
| cardboardbox02 | 6.4 deg | 57 mm | 47.3 deg | 125.2 deg | 183/220 |
| treasurebox05 | 4.5 deg | 126 mm | 33.2 deg | 63.6 deg | 40/217 |
| treasurebox02 | 2.0 deg | 105 mm | 77.5 deg | 96.1 deg | 136/263 |
| **median of 13** | **2.5 deg** | **47 mm** | **3.3 deg** | **34.0 deg** | |

Laptops and two of the books are measured to a few degrees. Box lids are where it fails: in
these 640 x 480 videos SAM loses the lid (and often the box) on most frames, even after the
agent re-anchors it, so the closed/open state is guessed from the whole box. Angle errors are
taken modulo 360 deg. Per-sequence results: `scripts/rbo_bench.py` writes `data/rbo/results/summary.md`.

### Open weights only

`run_agent.py --open-weights` (Muse Spark Open on Baseten + SAM 3.1 / SAM 3 on Modal):

- SAM box tracking on the synthetic lamp: 120/120 frames, IoU against the hosted API 0.970 /
  0.952, arm against ground truth 0.942 (hosted 0.960).
- Muse Spark Open driving the LiDAR laptop (SAM via the API): lid 312 x 208 mm, base 13 mm,
  0..117 deg, hinge on the real hinge, 9 turns.
- **Everything on open weights** (Muse Spark Open + SAM on Modal), LiDAR laptop from scratch,
  18 turns: lid 305 x 209 mm (13" MacBook: 304 x 212), opening 3..122 deg following the
  video (closes, then reopens), lid replay covering 91% of its masks. The base was segmented in
  fragments (keyboard, trackpad), so its thickness could not be measured; the pipeline and
  Spark's report both flag it instead of passing on a wrong number.

## Run it

Requires macOS or Linux, [uv](https://docs.astral.sh/uv/), ffmpeg. No local GPU.

```bash
uv sync                                  # Python 3.12 environment in .venv
cp .env.example .env                     # then fill in the keys you have
.venv/bin/python scripts/smoke_test.py   # one cheap call per backend
```

### A 2D phone video

Film on a tripod, camera facing the plane the part moves in, an ID-1 sized card in view,
open and close the object slowly.

```bash
.venv/bin/python scripts/run_agent.py my_video.mov --out data/tracks
open data/tracks/my_video/replay.mp4               # the model over the video
cat  data/tracks/my_video/agent/report.md          # Spark's report
ls   data/tracks/my_video/{model.urdf,model.xml}
```

Without the agent: `scripts/track_video.py VIDEO "laptop lid" "laptop keyboard"`, then
`scripts/fit_model.py VIDEO --ref "laptop lid=0.212"` (a known length instead of a card).

### An iPhone LiDAR capture

Record with [Stray Scanner](https://apps.apple.com/app/stray-scanner/id1557051662), export
the folder, then:

```bash
.venv/bin/python scripts/run_agent.py data/real/laptop_lidar --out data/tracks_agent
open data/tracks_agent/laptop_lidar/replay3d.mp4   # the model rendered from the phone's pose
```

### Robot demo and viewer

```bash
scripts/fetch_assets.sh                            # Franka Panda from MuJoCo Menagerie
.venv/bin/python scripts/robot_demo.py data/tracks_agent/laptop_lidar --out demo.mp4
.venv/bin/mjpython -m mujoco.viewer --mjcf=data/tracks_agent/laptop_lidar/model.xml   # interactive

.venv/bin/python scripts/export_viewer.py \
    laptop_lidar:data/tracks_agent/laptop_lidar:data/real/laptop_lidar/laptop_lidar.mp4
.venv/bin/python -m http.server 8931 --directory docs   # http://localhost:8931
```

The viewer (`docs/index.html`, three.js, no build) plays the phone video, the replay and the
robot demo next to the measured model in 3D, driven by the video's time, with joint sliders
and URDF/MJCF downloads.

### Open weights only (for the Open Weight prize check)

```bash
# 1. Muse Spark Open on Baseten: put BASETEN_API_KEY in .env.
# 2. SAM on your own Modal GPU: accept the SAM License on huggingface.co/facebook/sam3.1
#    and huggingface.co/facebook/sam3, then
.venv/bin/modal setup
.venv/bin/modal secret create huggingface HF_TOKEN=hf_...
.venv/bin/modal deploy modal_apps/sam3_open.py
.venv/bin/python scripts/check_sam_open.py --e2e          # synthetic lamp vs the hosted tracks
# 3. Run everything on open weights:
.venv/bin/python scripts/run_agent.py data/real/laptop_lidar --open-weights --out data/tracks_open
```

`--sam api` keeps the hosted segmenter while reasoning runs on open weights. The Modal app
scales to zero after 2 idle minutes (A10G, about $0.05-0.10 per video).

### The RBO benchmark

```bash
.venv/bin/python scripts/rbo_download.py laptop01 laptop02 ...   # into data/rbo/ (not committed)
.venv/bin/python scripts/rbo_bench.py                            # agent + scoring, all downloaded sequences
cat data/rbo/results/summary.md
```

## Limitations

- v1 handles 1-3 joints. The 3D path (`fit_lidar`) handles one hinged plate or one slide.
- Thin parts make replay IoU harsh; book covers and box lids track poorly (see the benchmark).
- A 2D side view cannot measure the width across the motion plane; it is assumed or set.
- Floor levelling removes ARKit drift in height and tilt, not along the floor.

## Built on

Models: Muse Spark, Muse Spark Open, SAM 3.1, SAM 3 (Meta).

| Project | License | Used for |
|---|---|---|
| [MuJoCo](https://github.com/google-deepmind/mujoco) | Apache-2.0 | simulation, rendering the replay |
| [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) (Franka Panda) | Apache-2.0 | robot demo, fetched by `scripts/fetch_assets.sh`, not committed |
| [facebookresearch/sam3](https://github.com/facebookresearch/sam3) | SAM License | open-weights SAM on Modal |
| [meta-sam-parser](https://pypi.org/project/meta-sam-parser/) | SAM License | parsing SAM API output; `video2urdf/fast_mask.py` is a numba port of its decoder (`LICENSES/SAM_LICENSE`) |
| [RBO dataset](https://zenodo.org/records/1036660) - R. Martín-Martín, C. Eppner, O. Brock, "The RBO Dataset of Articulated Objects and Interactions", IJRR 2019 | CC BY 4.0 | benchmark; downloaded, not redistributed |
| [three.js](https://threejs.org) | MIT | web viewer (CDN) |
| [mapbox-earcut](https://github.com/skogler/mapbox_earcut_python) | ISC | triangulating mesh caps |
| OpenCV, NumPy, SciPy, numba, OpenAI Python SDK (as the OpenAI-compatible client), Modal | various OSS | |
| [Stray Scanner](https://github.com/strayrobots/scanner) | app | recording iPhone LiDAR captures |

What we built on top: the agent and its tools, the SAM tracking strategy (key-frame concept
prompts, then multi-anchor box tracking forward and over a reversed clip), the 2D and 3D joint
fits, link geometry, URDF/MJCF emission, the camera-pose MuJoCo replay and its scoring, the
robot demo, the viewer and the RBO benchmark harness.
