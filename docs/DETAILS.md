# Video2URDF: details

More results and commands, moved here from the [README](../README.md) to keep it short.
Numbers are as of the commits that added them; the README's tables are kept in step with the
published models in the viewer.

## More results

### Repeatability: the agent three times on the same laptop

Three fresh Muse Spark runs on the LiDAR laptop (`data/repeat/`, 13-19 turns each, all ended
with a model and physics): lid 316 x 216, 304 x 215, 304 x 214 mm (spec 304 x 212). In all
three SAM's screen track slid onto the lid's back as it shut (52-77% of the frames that show
both); re-tracking doesn't change that, so `fit_lidar` now measures those frames from the
closed-lid part instead. The same tracks refit with that: lid 316 x 217 / 305 x 209 /
303 x 212 mm, lid replay 75 / 95 / 92%, base 86 / 86 / 31% (run 3's base candidate was the
whole laptop; the tool flags it). The base candidate is what still varies between runs.
After tightening reference selection, a separate refit of the same run 2 and 3 tracks at
quarter-resolution replay improved base mask coverage from 15.6% to 86.0% and 31.1% to
95.0% respectively. This is not a new independent run; one refit produced a suspect 24 mm
lid thickness and remains flagged. Coverage is computed only on the usable tracked frames,
not against independent ground truth. See [validation details](appearance_validation.md).

### Rods on hinges in 3D, and objects the phone walked around

- **A desk lamp from iPhone LiDAR** (`video2urdf/chain3d.py`, agent tool `fit_chain`). At 256x192
  depth a rod is 2-3 pixels wide, ARKit marks it low confidence and every pixel sits on a depth
  edge, so the plate and point-tracking fits see nothing. Each frame, each rod's raw depth (the
  wall behind its edge dropped) gets a 3D line, held in the plane its image centre line spans; a
  hinge is the plane the rod's direction turns in, at the rod's near end, and a child rod is
  measured after undoing its parent's turn. Synthetic oblique lamp: axis 0.4 / 4.8 deg, angle
  0.06 / 1.1 deg mean. Real lamp (1385 frames, Spark chose `fit_chain` itself): Spark fit both
  rods as hinges (pole 294 mm, arm 370 mm, elbow -8..77 deg) and reported the pole's motion as depth
  noise, to be treated as fixed. We refit the same tracks with the pole fixed (`fixed=rod1`, our
  step, not the agent's): pole 351 mm, arm 380 mm, 2..89 deg; the replay covers 63% / 76% / 31% of
  the base's, pole's and arm's masks.
- **A black mouse the phone circled** (`fit_static`). Its few confident depth points made a
  59 x 27 x 25 mm box. When the phone went more than 60 deg around, the object is now carved
  from 48 views' masks (a visual hull) and capped at the depth-measured top: 121 x 75 x 55 mm
  (MX Master 3S: 125 x 84 x 51), covering 88% of its masks. Spark judged 0.14 kg (spec 141 g).
- **SAM 3D Objects on any fitted model**: placed by the replay's camera poses and joint values,
  and for static objects and rod chains judged like the shape design (over the video, within 0.10
  of the measured meshes) plus the depth check. The mouse's mesh: the depth points seen on its
  frame lie 0.5 mm from it (median; one direction only: the mesh's unseen side is generated, and
  its points lie 32 mm from those depth points, median), mask F-score 0.89 vs 0.94; it is the
  mouse's visual in the viewer.

### The robot works the object: grasp, open, close, carry

`scripts/robot_tasks.py` puts the measured object in front of a Franka Panda (MuJoCo Menagerie)
and picks tasks from what was measured: a hinged plate is pinched at its free edge, closed and
opened again; a hinged rod (a lamp's arm) is pinched near its tip and swung to the far end of its
measured range and back; a drawer is pulled out by the handle Shape design drew (its collision is
switched on in this scene) and pushed back in; an object without joints is picked up, carried
aside, then brought back turned a quarter. The gripper's own actuator squeezes (~30 N, stiffer
than Menagerie's default, which holds ~1 N on an 18 mm handle; the real Panda Hand holds 70 N) and
the object moves only through finger contact; arm joints get gravity compensation. Each video
titles its tasks and shows the joint value the object actually reaches. Results: LiDAR laptop
closed 87 deg and reopened; drawer out 199 mm and back to 0 mm; LiDAR lamp's arm lowered 40 deg
and raised back; mouse carried 18 cm and returned turned 90 deg; side-view laptop closed and
reopened; side-view lamp's arm raised 33 deg and lowered back. A side-view rod is measured as a
6 mm silhouette, too thin to hold, so there the fingers hold the rod Shape design drew (14 x 9 mm),
and joints not being worked are held by friction. The drawer and the cabinet keep their measured
collision (the cabinet has an open slot, the drawer is an open tray): the pads hold the handle's
outer part with the fingertips 2 mm off the drawer's front, and the hand's deepest contact with
anything but the handle is 1.2 mm, MuJoCo's soft contact. `robot_tasks.py` prints that depth for
every part the hand touches.

```bash
.venv/bin/python scripts/robot_tasks.py data/tracks_agent2/mouse_lidar   # -> robot_tasks.mp4 in the run
```

### Faster tracking, and scale without a reference

- **SAM tracking ~1.8x faster.** The forward pass and the backward pass (a reversed clip) are
  independent requests: they now run at once, on two Modal GPU containers for the open weights
  (`track_pass`) or two Meta API requests. Videos above 40 fps are tracked on every second frame
  (`SAM_TRACK_STRIDE`, default auto) and the frames between take the nearest tracked mask.
  LiDAR lamp, 1385 frames, 4 parts, SAM 3.1/3 on Modal: 520 s -> 285 s; masks against the old run:
  IoU 0.995 / 0.993 (table, base), 0.98 / 0.97 (the two thin rods; p10 0.92 / 0.81).
- **Automatic scale for ordinary video** (`video2urdf/autoscale.py`, run by `video_to_model.py`
  when no `--scale`/`--reference` is given). MapAnything's scale can be several times off, and its
  depth stretches lengths along the line of sight. An ID-1 card found by SAM 3.1 is measured by
  its edge across the view (85.6 mm, or 54 mm); without one, Muse Spark names the object and its
  usual largest size and the object is measured on frames where it is wholly in view with its
  long axis across the view. The card wins when it lies about as far away as the object.
  Checks: on `notebook_2` the card gives x0.75, consistent over 8 frames (p90/p10 1.09); the
  laptop is cut off by the frame there, so no size-based scale is taken (it read x1.21 before
  that check). On the RGB copy of the LiDAR laptop, already at LiDAR scale, the object's usual
  size gives x1.01 (expected 1.0).

### The model looks like the object: painted with its own video

The measured shapes are boxes and outlines; `video2urdf/texture.py` paints them with the video.
Each flat face of each link takes the frame where it was seen largest and most head-on, not
hidden behind another part of the model (MuJoCo renders, from the phone's pose on that frame,
which face is in front) nor behind a hand (the point must be inside that link's SAM mask). The
face is rectified by a homography into one texture atlas; pixels outside the link's mask and
faces thinner than 1 cm take the link's colour. On the LiDAR laptop the screen shows its
wallpaper, the base its keyboard and trackpad, the shut lid the logo seen while it was
closed; the drawer's front its wood grain and handle. Shapes, and so collisions and inertia,
are unchanged. The viewer, the robot demo and the MJCF/URDF (textured OBJ visuals) use it.

## More commands

### Any phone video, no LiDAR (MapAnything, experimental)

[MapAnything](https://github.com/facebookresearch/map-anything) (Meta, Apache-2.0 weights
`facebook/map-anything-apache`) predicts metric depth, intrinsics and camera poses from RGB
alone. `scripts/video_to_capture.py` runs it on Modal over chunks of the video that share
anchor frames, stitches the chunks with a similarity fitted on the anchors' points, and writes
the result in the Stray Scanner layout, so the LiDAR path runs on it unchanged:

```bash
.venv/bin/modal deploy modal_apps/mapanything_open.py            # once; A10G, scales to zero
.venv/bin/python scripts/video_to_capture.py my_video.mov --focal-35mm 26   # iPhone 1x is ~26 mm
.venv/bin/python scripts/run_agent.py data/real/my_video_mapanything --sam open
# sizes off? rescale the capture by one known length (no new inference), then fit again:
.venv/bin/python scripts/video_to_capture.py my_video.mov --scale KNOWN_M/MEASURED_M
```

Checked against the LiDAR laptop by running its RGB alone (`scripts/eval_pseudo_depth.py`,
`scripts/compare_hinges.py`, same SAM tracks): MapAnything's metric scale was 3.6x too large on
this close top-down shot, but steady over the video (p10-p90 within +-4%), so one known length
can correct the overall scale. It does not eliminate local depth or camera errors. After that,
depth is within 5% of the LiDAR (median), camera rotation within 2 deg of
ARKit (median), and the hinge axis direction within 2.0-2.3 deg of the LiDAR fit; the opening
angle is off by 8-12 deg (median) and the base thickness is not measurable. It can provide an
approximate hinge estimate when the reconstruction is good; use measured depth for more reliable
dimensions. Film obliquely, as for LiDAR: on a
side view (`notebook_2.MOV`, no LiDAR) the agent put the hinge on the real hinge, but the lid is
seen edge-on, its plane fits were poor (replay covers 13% of the lid) and Spark said so and
asked for a reshoot; side views go through the 2D path instead.

These are aggregate results for earlier fits, not a guarantee for each region. A later
same-video experiment found opposite local depth errors on the floor and laptop base, with
a resulting 78 mm base thickness and badly undersized lid despite global scale adjustment.
See [validation details](appearance_validation.md) before relying on RGB-only dimensions.

### Robot demo and viewer

After a fit (the agent's run does physics itself):

```bash
.venv/bin/python scripts/texture_model.py data/tracks_agent/laptop_lidar data/real/laptop_lidar/laptop_lidar.mp4
.venv/bin/python scripts/physics_demo.py data/tracks_agent/laptop_lidar
```

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

### Realistic link meshes (SAM 3D Objects)

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
additional time. SAM 3D is also optional.

For a plain video, deploy `modal_apps/mapanything_open.py` first and pass the video path to
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

### URDF Doctor

```bash
.venv/bin/python scripts/urdf_doctor.py examples/doctor/laptop_buggy.urdf data/tracks_agent/laptop_lidar
.venv/bin/python scripts/urdf_doctor.py examples/doctor/laptop_spec.urdf data/tracks_agent/laptop_lidar \
    --joint-log examples/doctor/joint_log_flipped.csv
```

### The RBO benchmark

```bash
.venv/bin/python scripts/rbo_download.py laptop01 laptop02 ...   # into data/rbo/ (not committed)
.venv/bin/python scripts/rbo_bench.py                            # agent + scoring, all downloaded sequences
cat data/rbo/results/summary.md
```

