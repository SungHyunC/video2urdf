# Commands

Every command, by task. Requires macOS or Linux, [uv](https://docs.astral.sh/uv/), ffmpeg. No
local GPU. The README has the two quickest runs: [Reproduce it](../README.md#reproduce-it) (the
sample capture on the Meta Model API) and
[Open weights only](../README.md#open-weights-only-for-the-open-weight-prize-check) (also Muse
Glimmer on your own GPU, or any OpenAI-compatible server). Each script's docstring has its
usage; [scripts/README.md](../scripts/README.md) lists the scripts by purpose. What each step
gave: [RESULTS.md](RESULTS.md).

- [Setup](#setup)
- [Any capture, one command](#any-capture-one-command)
- [A 2D phone video](#a-2d-phone-video)
- [An iPhone LiDAR capture](#an-iphone-lidar-capture)
- [Any phone video, no LiDAR (MapAnything, experimental)](#any-phone-video-no-lidar-mapanything-experimental)
- [Shape design](#shape-design)
- [Texture, physics check, robot tasks and the viewer](#texture-physics-check-robot-tasks-and-the-viewer)
- [Completing an unseen body](#completing-an-unseen-body)
- [Realistic link meshes (SAM 3D Objects)](#realistic-link-meshes-sam-3d-objects)
- [Experimental: object-aware appearance completion](#experimental-object-aware-appearance-completion)
- [URDF Doctor](#urdf-doctor)
- [The RBO benchmark](#the-rbo-benchmark)
- [The drawer policy](#the-drawer-policy)
- [The no-tools baseline](#the-no-tools-baseline)
- [Check the downloads](#check-the-downloads)
- [Tests](#tests)

## Setup

```bash
uv sync                                  # Python 3.12 environment in .venv
cp .env.example .env                     # then fill in the keys you have
.venv/bin/python scripts/smoke_test.py   # one cheap call per backend
```

## Any capture, one command

```bash
.venv/bin/python scripts/video2urdf.py INPUT [--robot] [--open-weights]   # -> data/runs/<name>/
.venv/bin/python scripts/video2urdf.py INPUT --check-only                   # what it would do, nothing billed but one look
```

`INPUT` can be a Stray Scanner zip or folder (iPhone LiDAR), any phone video (iPhone MOV with
HEVC, HDR, rotation or variable frame rate; Android MP4; 60 or 120 fps), an RGB-D folder, a photo
folder, or a zip of any of these; the type is read from the content. Plain video is turned
upright, tone-mapped to SDR and brought to a steady 30 fps or less; depth captures keep their
frames aligned with their depth. A quick capture check prints reshoot tips. A depth capture
takes the 3D path; a plain video goes 3D through MapAnything when the camera moves or looks down
at an angle, and the planar path when a still camera faces the motion (`--path` overrides).
Then the agent, physics, textures, Spark's shape design, completing an unseen body, the robot
tasks (`--robot`) and a viewer folder, with `summary.md` saying what was measured, what was
estimated and the status. If no joint is found it still writes a rigid model, marked
`needs_review`. What it gave on seven test inputs: [RESULTS.md](RESULTS.md#one-command-on-seven-inputs).

The last step loads the final downloads (the viewer's `model.zip`) as a user would, with
`scripts/check_exports.py` (MuJoCo; PyBullet, yourdfpy and Drake when installed, else
`not_tested`), and writes `validation.json` with a row per check and the model files' hashes.
`summary.json` keeps `agent_status` and `artifact_status` apart; `status` is the worse of them,
and `summary.md` names each failing or warning check in one line. A failing format lowers only
the artifact status; the model and its other formats stay. After changing a run's model:

    .venv/bin/python scripts/validate_run.py data/runs/NAME            # check again, update the summary
    .venv/bin/python scripts/validate_run.py data/runs/NAME --status   # is validation.json current?

Other options: `--out DIR`, `--name N`, `--viewer-out DIR`, `--glimmer`, `--sam api|open`,
`--path auto|3d|planar`, `--rotate auto|0|90|180|270`, `--no-look`, `--no-design`,
`--no-texture`; for a plain video on the 3D path `--focal-35mm F`, `--scale K`,
`--no-auto-scale`; caps `--max-fps`, `--max-frames`, `--max-side`, `--image-fps`
(`scripts/video2urdf.py --help`).

The sections below run the steps one at a time.

## A 2D phone video

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

## An iPhone LiDAR capture

Record with [Stray Scanner](https://apps.apple.com/app/stray-scanner/id1557051662), export
the folder, then:

```bash
.venv/bin/python scripts/run_agent.py data/real/laptop_lidar --out data/tracks_agent
open data/tracks_agent/laptop_lidar/replay3d.mp4   # the model rendered from the phone's pose
```

The agent ends with `set_physics` (masses, friction). For a run made without it:
`.venv/bin/python scripts/add_physics.py data/tracks_agent/laptop_lidar` (Spark watches the
video once, structured output; `--reuse` re-applies the saved estimate).

## Any phone video, no LiDAR (MapAnything, experimental)

[MapAnything](https://github.com/facebookresearch/map-anything) (Meta, Apache-2.0 weights
`facebook/map-anything-apache`) predicts metric depth, intrinsics and camera poses from RGB
alone. `scripts/video_to_capture.py` runs it on Modal over chunks of the video that share
anchor frames, stitches the chunks with a similarity fitted on the anchors' points, and writes
the result in the Stray Scanner layout, so the LiDAR path runs on it unchanged:

```bash
.venv/bin/modal deploy modal_apps/mapanything_open.py            # once; A10G, scales to zero
.venv/bin/python scripts/video_to_capture.py my_video.mov   # focal from the file (iPhones: 25 mm), else --focal-35mm
.venv/bin/python scripts/run_agent.py data/real/my_video_mapanything --sam open
# sizes off? rescale the capture by one known length (no new inference), then fit again:
.venv/bin/python scripts/video_to_capture.py my_video.mov --scale KNOWN_M/MEASURED_M
```

How close it gets to LiDAR on the same object: [RESULTS.md](RESULTS.md#plain-video-without-lidar-mapanything).

`scripts/video_to_model.py` takes a plain video too. For a plain video, deploy `modal_apps/mapanything_open.py` first and pass the video path to
the same command. MapAnything predicts depth and an uncertain scale, not LiDAR: calibrate
with a known length (`--scale KNOWN_LENGTH/MEASURED_LENGTH`, evaluated as a number) and
independently check dimensions. Depth and camera errors remain after scale calibration.
For a new video capture, `--reference FRAME X1 Y1 X2 Y2 LENGTH_MM` instead calculates that
scale from the estimated depth at two RGB-frame pixels whose real distance is known. Pick
both points on visible surfaces, away from depth edges. This is one global scale correction,
not a cure for depth or focal-length errors; the points and correction are recorded in
`mapanything.json` and `measurement_source.json`. Add repeatable
`--check-reference FRAME X1 Y1 X2 Y2 LENGTH_MM` arguments for independently measured
lengths: these are not used to fit scale, and their residual errors are recorded. Errors
over 5% are flagged for review; passing checks at a few points does not validate all depths.
To correct an already reconstructed MapAnything capture without rerunning depth inference
or modifying the original, make a new calibrated copy (replace the example pixels and
lengths with your own measurements):

```bash
.venv/bin/python scripts/calibrate_capture.py data/item_mapanything data/item_calibrated \
  --reference 8 245 115 245 225 148.1 --check-reference 8 240 140 285 140 52.9
.venv/bin/python scripts/video_to_model.py data/item.mp4 --capture-out data/item_calibrated \
  --out data/item_run --viewer-out data/item_viewer
```

The new path must not exist. The points and lengths must describe visible rigid surfaces;
the check uses a different pair and is not a substitute for measuring the finished model.
The optional viewer export writes a self-contained ZIP with URDF, MJCF, meshes and textures;
serve it locally as printed by the command. Use new output directories so earlier results
are not overwritten.

## Shape design

`run_agent.py` runs Spark's shape design at the end of every run (`--no-design` skips it). By hand:

```bash
.venv/bin/python scripts/design_shape.py data/tracks/my_video my_video.mov   # run_agent.py does this by default
.venv/bin/python scripts/design_shape.py data/tracks/my_video --remove        # back to the measured visuals
```

## Texture, physics check, robot tasks and the viewer

After a fit (the agent's run does physics itself):

```bash
.venv/bin/python scripts/texture_model.py data/tracks_agent/laptop_lidar data/real/laptop_lidar/laptop_lidar.mp4
.venv/bin/python scripts/physics_demo.py data/tracks_agent/laptop_lidar
```

The robot tasks (a Franka Panda grasps the measured object and works it; what it does depends on
what was measured: [RESULTS.md](RESULTS.md#the-robot-works-the-object-grasp-open-close-carry)):

```bash
scripts/fetch_assets.sh                            # Franka Panda from MuJoCo Menagerie
.venv/bin/python scripts/robot_tasks.py data/tracks_agent2/mouse_lidar   # -> robot_tasks.mp4 in the run
.venv/bin/python scripts/robot_demo.py data/tracks_agent/laptop_lidar --out demo.mp4   # the earlier push-only demo
.venv/bin/mjpython -m mujoco.viewer --mjcf=data/tracks_agent/laptop_lidar/model.xml   # interactive
```

The viewer:

```bash
.venv/bin/python scripts/export_viewer.py \
    laptop_lidar:data/tracks_agent/laptop_lidar:data/real/laptop_lidar/laptop_lidar.mp4
.venv/bin/python -m http.server 8931 --directory docs   # http://localhost:8931
```

The viewer (`docs/index.html`, three.js, no build) plays the phone video, the replay and the
robot demo next to the measured model in 3D, driven by the video's time, with joint sliders
and URDF/MJCF downloads.

## Completing an unseen body

For a door model whose body is only the part SAM tracked (a microwave's control-panel strip):
the front measured, the hidden depth bounded by the video, the rest estimated by Muse Spark
([what it gave](RESULTS.md#the-parts-the-camera-never-saw)). Without `--force` it only acts on
runs whose body is known to be partial.

```bash
.venv/bin/python scripts/complete_body.py RUN VIDEO [--capture DIR] [--check | --remove] \
    [--force] [--open-weights] [--prompt NOUN] [--no-spark] [--note TEXT]
```

`scripts/complete_body.py RUN VIDEO --remove` restores the measured-only model.

## Realistic link meshes (SAM 3D Objects)

```bash
# Request access on huggingface.co/facebook/sam-3d-objects (gated), then
.venv/bin/modal deploy modal_apps/sam3d.py
.venv/bin/modal run modal_apps/sam3d.py::download     # 12 GB of checkpoints into a Modal volume
.venv/bin/python scripts/sam3d_meshes.py data/tracks_agent/laptop_lidar \
    --video data/real/laptop_lidar/laptop_lidar.mp4 --capture data/real/laptop_lidar
```

Writes candidate meshes and `sam3d.json` (depth and size checks). Accepted candidates become
`meshes/<link>_sam3d.{obj,ply}` and visuals in `model.xml` / `model.urdf`; rejected candidates
leave the prior visual in place. In
MJCF the measured shapes move to geom group 3 (hidden in the viewer, still colliding). L40S,
scales to zero after 2 idle minutes.

For SAM 3D meshes add `--sam3d` to `video_to_model.py`. Each mesh is checked against observed
depth points; a part whose median distance exceeds 5 mm or p90 exceeds 15 mm is saved for
inspection but not activated. Passing this check does not validate unobserved surfaces.
On the LiDAR laptop and drawer (10/31) every candidate was rejected: the meshes matched the seen
surfaces (0.8-2.9 mm) but came out far too big, e.g. a lid 81 mm thick and 332 x 245 mm across
(measured about 309 x 205 x 4 mm), so the measured visuals stay. Flat, thin parts are where SAM 3D
inflates the unseen side.
We also require each generated extent to be within 0.8–1.2 times the fitted extent; fitted
dimensions can themselves include assumptions, especially for hidden drawer depth. Static
objects and rod chains, whose fitted extent is an envelope (a visual hull, a rod) rather than a
size to match, skip this ratio and are judged over the video instead (the mouse's mesh is
1.04 / 1.11 / 1.27 times its hull).
SAM 3D always gets the captured or MapAnything depth as its pointmap. Without one it would
estimate depth with MoGe, which is not a Meta model, so that option was removed.
See [appearance validation](appearance_validation.md) for the tested captures and
examples of generated meshes that failed the checks.

## Experimental: object-aware appearance completion

None of the published viewer models use this path; its log is
[appearance_validation.md](appearance_validation.md).

Object-aware appearance completion: Spark identifies the object and likely materials from
the video. By default it uses a simple visual; with `--video-texture`, observed faces can
use projected video colours and unseen faces get a synthetic material. Laptop base
and lid get straight-edged inferred panels on the fitted outline; the drawer builder completes a
cabinet and open tray with assumed hidden dimensions. The drawer template also adds
plausible visual-only fronts for other closed drawers and handles when visible; their
exact positions are guessed. If the visible drawer count is uncertain, the visual assumes
three drawers and flags that assumption in the quality report. For a book, the visible inner cover
and hinge define a hypothesized rectangular outer cover. These templates can be wrong and
leave the fitted collisions unchanged. For an apparently stationary rigid box or upright
cylinder, the RGB-D agent can generate a simple single-link shape from visible depth. Its
hidden sides and any circular cross-section are assumptions; `quality.json` flags them.
The stationarity check can mistake viewpoint or mask changes for motion, and cylinder
fitting has only been tested on synthetic depth. Other object types retain the fitted shape;
arbitrary objects and RGB-only metric dimensions are not validated. A single
command can fit an RGB-D capture and complete its appearance:

```bash
.venv/bin/python scripts/video_to_model.py data/real/laptop_lidar --out data/new_run --viewer-out data/new_viewer
```

For an articulated object filmed approximately from the side, an explicit planar path
is also available: `.venv/bin/python scripts/video_to_model.py data/real/stand.mov --planar --out data/stand_run --viewer-out data/stand_viewer`.
It classifies the object and uses a side-view silhouette model; hidden depth, thickness,
and possibly metric scale are assumed. Classification and the resulting shape must be
reviewed. This path does not use RGB depth reconstruction, SAM 3D, or video texturing.

This default creates a recognizable simplified visual and retains the fitted geometry and
articulation. Add `--video-texture` to project video colours; that optional step takes
additional time. SAM 3D is also optional. A plain video input and its scale options:
[above](#any-phone-video-no-lidar-mapanything-experimental).

For an accepted SAM 3D visual, an experimental manual texture transfer can be run on a
new copy. Select a textured fitted mesh and a replay frame where that particular surface
is visible; repeat `--view` for distinct sides. For the local laptop experiment:

```bash
.venv/bin/python scripts/transfer_sam3d_texture.py data/appearance_trials/sam3d_alternate_open data/real/laptop_lidar/laptop_lidar.mp4 --out data/appearance_trials/my_texture_trial --link lid --view meshes/screen_tex.obj 150 --view meshes/lid_infer_tex.obj 343
```

For a new run, `scripts/video_to_model.py ... --sam3d --sam3d-video-texture` requests
the same experimental automatic transfer when a SAM 3D link passes geometric checks.
Add `--sam3d-retries 2` to try up to two more temporally separated frames after a
rejection; this uses additional remote inference and records attempts in `sam3d_attempts.json`.
Alternatively use `--auto-views` instead of `--view` to select only faces and frames
recorded as video-observed by the texture stage. Both modes are experimental; inspect
the generated texture on all sides. The 25 mm default projection tolerance is for
appearance only, and the generated shape, occlusions, texture edges and hidden surfaces
need review; it is not a measurement. The result remains marked for review.

For an existing successful 3D fit, run appearance completion separately:

```bash
.venv/bin/python scripts/infer_appearance.py data/tracks_agent/laptop_lidar data/real/laptop_lidar/laptop_lidar.mp4
```

`appearance.json` records the model's classification and material guesses; `texture.json`
identifies synthetic faces and `shape_inference.json` records the shape hypothesis.
Measured collision geometry and joint physics do not change. These synthetic surfaces and
shapes are guesses, not observations of hidden features. `--reuse` reuses the saved
classification. The classification call uses the configured Spark API and sends the video
(Glimmer uses sample frames). These templates do not reconstruct arbitrary object classes.
Use `--simple` on this standalone command to match `video_to_model.py`'s simple-colour
default. It can also resume appearance after an interrupted fit, but doing so is a resumed
run rather than evidence that the complete automatic pipeline succeeded.
To resume on a separate copy while explicitly reusing a prior classification:

```bash
.venv/bin/python scripts/resume_appearance.py FIT_DIR VIDEO.mp4 NEW_DIR \
  --appearance-from PRIOR_APPEARANCE.json --capture ORIGINAL_CAPTURE_DIR
```

`--capture` is needed when the interrupted fit has no `measurement_source.json`.
Omit `--appearance-from` to make a new classification call. The result records its
resumed provenance and the source directory is left unchanged.
New visual geometry is compared with the fitted model on sampled tracked frames; if coverage
or, for a full-part mask, IoU falls by more than five percentage points or there are fewer than five usable frames, the
new geometry is not activated. See `shape_inference.json` for the comparison and decision.

## URDF Doctor

```bash
.venv/bin/python scripts/urdf_doctor.py examples/doctor/laptop_buggy.urdf data/tracks_agent/laptop_lidar
.venv/bin/python scripts/urdf_doctor.py examples/doctor/laptop_spec.urdf data/tracks_agent/laptop_lidar \
    --joint-log examples/doctor/joint_log_flipped.csv
```

What it found on these files: [RESULTS.md](RESULTS.md#urdf-doctor-checking-an-existing-urdf-against-the-video).

## The RBO benchmark

```bash
.venv/bin/python scripts/rbo_download.py laptop01 laptop02 ...   # into data/rbo/ (not committed)
.venv/bin/python scripts/rbo_bench.py                            # agent + scoring, all downloaded sequences
cat data/rbo/results/summary.md
```

`scripts/rbo_bench.py --open-weights` runs the agent on Muse Spark Open + SAM weights and keeps
those runs apart from the Meta API ones.

## The drawer policy

PPO on the measured drawer, trained on Modal CPUs
([what it learned](RESULTS.md#a-policy-learned-on-the-measured-drawer)):

```bash
uv sync --extra rl
.venv/bin/modal run modal_apps/rl_drawer_train.py::main --steps 5000000 --envs 32 --name drawer_ppo_honest
.venv/bin/modal run modal_apps/rl_drawer_train.py::arm --name drawer_ppo_honest --seeds 20
.venv/bin/python scripts/run_drawer_policy.py data/tracks_design/drawer2_lidar --seed 20012   # the arm video
```

## The no-tools baseline

The same Muse Spark asked plainly for a URDF of each video (Meta Model API key; about 12k
tokens per answer). Commands, scoring and all 30 answers: [spark_alone.md](spark_alone.md#files).

## Check the downloads

The viewer's "Download model" zip holds `model.urdf`, `model.xml` and their meshes. To check
them the way a user loads them (MuJoCo for both files, PyBullet, yourdfpy, Drake):

```bash
uv run --no-project --python 3.12 --with pybullet --with yourdfpy --with mujoco --with drake \
    python scripts/check_exports.py docs/models     # one row per model per loader
```

On Apple Silicon pybullet has no wheel, so uv builds it from source the first time (a few
minutes). Simulate the MJCF in MuJoCo (it carries the contact exclusions a URDF can't); ROS
users need to rewrite the mesh paths as `package://`.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -q
```
