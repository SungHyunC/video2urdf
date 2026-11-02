# Reproduce the published numbers: no API keys, no GPU

The iPhone LiDAR laptop in the viewer (`docs/models/laptop_lidar`, the README's 309 x 205 mm lid)
was measured by the agent: Muse Spark chose the parts and SAM 3.1 tracked them, then the fit turned
the tracks into the model. The model calls are the part that needs keys and credits, so their
results are saved: SAM 3.1's tracks, the parts Spark tracked, the role it gave each in the fit and
its physics estimate, 1.9 MB in all. `scripts/reproduce.py` reruns everything after them on your
CPU and prints the published numbers next to the reproduced ones. It reads no `.env`, calls no
model or API, and refuses network connections once its two downloads are done.

## Run it

```bash
git clone https://github.com/SungHyunC/video2urdf && cd video2urdf
uv sync                                  # Python 3.12 into .venv
.venv/bin/python scripts/reproduce.py    # about 5 minutes on an M1 MacBook (below)
```

It downloads two release assets into `data/reproduce/downloads/` and checks them:

| file | release | size | what |
|---|---|---|---|
| `laptop_lidar_stray.zip` | `sample-data-v1` | 39 MB | the Stray Scanner capture behind the README's laptop, as published (sha256 `0ee98f44...`) |
| `laptop_lidar_tracks.zip` | `sample-data-v2` | 1.9 MB | the saved model decisions for that capture (sha256 `962b9b4a...`) |

Both releases are published. If a download fails (no network, a moved file), the command stops
right away and says what to do, for example:

```
Stopped: the saved tracks aren't published yet: https://github.com/SungHyunC/video2urdf/releases/download/sample-data-v2/laptop_lidar_tracks.zip answered HTTP 404 (the sample-data-v2 release doesn't exist yet). If you have laptop_lidar_tracks.zip, pass it with --bundle PATH.
```

With the files at hand, nothing is downloaded:

```bash
.venv/bin/python scripts/reproduce.py --bundle laptop_lidar_tracks.zip --capture laptop_lidar_stray.zip
```

(`--capture` also takes the unpacked capture folder, e.g. `data/real/laptop_lidar` from the
README's "Reproduce it".) Each run writes a new folder, `data/reproduce/laptop_lidar` (then
`_2`, ...): `model.urdf`, `model.xml`, `meshes/`, `replay3d.mp4` (the model drawn over the video
from the phone's own camera pose on every frame), `hinge.jpg`, `hinge.json`, `model3d.json`,
`physics.json`, `fit_lidar.txt` (what the fit tool said), `reproduce.md` (the table) and
`reproduce.json` (the table, timings, versions, the bundle's manifest).

## What is in the bundle

| file | what |
|---|---|
| `tracks.sam.txt` | SAM 3.1's masks for the five tracked parts on all 923 frames, in the Model API's text format (2.7 MB; decoded locally, which gives the same 3262 masks bit for bit as the 12 MB decoded cache the published fit read) |
| `parts.json` | the parts Muse Spark tracked: names, SAM prompts ("wooden floor", "silver laptop body", "laptop screen", "gray laptop lid", "black keyboard"), and the frames and boxes it anchored them on |
| `roles.json` | the role Spark gave each part in `fit_lidar`: floor, base, plate (the screen), plate_closed (the lid's back), open_indicator (the keyboard) |
| `physics_estimate.json` | Spark's material, mass and contact friction per link, and the spans where the lid stood still with no hand on it |
| `MANIFEST.json` | the files' sha256, and the capture they were tracked on (its zip's sha256, and those of its `rgb.mp4`, `odometry.csv` and `camera_matrix.csv`) |

All of it is derived from the published capture. `scripts/make_reproduce_bundle.py` builds it from
the run folder (fixed entry times and order: the same files give the same zip and sha256).

## What runs

1. A check that this machine can render with MuJoCo offscreen and write H.264 video (the replay
   needs both), before anything long.
2. Both zips unpacked and checked: every bundle file against the manifest, the capture against
   the one the tracks were made on. A missing or edited file stops the run with what to do; it
   never falls back to asking a model.
3. Offline from here: the API key variables are taken out of the process's environment, any
   internet connection or name lookup raises, and the agent's model clients are stand-ins that
   raise if anything uses them.
4. The agent's own `fit_lidar` tool (`Agent.tool_fit_lidar`, the call the agent makes) on the
   saved tracks and roles: the tracks decoded, the camera levelled against the floor, the lid's
   plane fitted from LiDAR depth on every frame, the hinge, the base and lid measured from depth,
   URDF and MJCF written, and the model rendered from the phone's camera pose on all 923 frames
   over the video, scored by how much of each SAM mask it covers.
5. `physics.apply_estimate` on Spark's saved estimate (what `scripts/add_physics.py --reuse`
   runs): inertia from the measured meshes at Spark's masses, the hinge's friction from the
   frames where the lid stood by itself, and the release check (the model let go in MuJoCo for
   2 s at five of those angles, with and without that friction).
6. The table.

Not rerun: Spark's choices and SAM's tracking (the bundle is their output), and what the viewer
adds on top of the measured model (Spark's shape design, the video texture, the robot demo).

## Expected output

From a fresh clone (commit `419e150`, `uv sync`, no `.env`), the bundle given as a local file
and the capture downloaded; paths shortened to `.`:

```
$ .venv/bin/python scripts/reproduce.py --bundle laptop_lidar_tracks.zip
Reproducing ./docs/models/laptop_lidar/model.json into ./data/reproduce/laptop_lidar.
No API key or model call is used; the network is refused once the downloads are done.

capture: downloading https://github.com/SungHyunC/video2urdf/releases/download/sample-data-v1/laptop_lidar_stray.zip
  10 of 39 MB
  20 of 39 MB
  30 of 39 MB
saved tracks: ./data/reproduce/downloads/bundle/laptop_lidar_tracks (made from data/tracks_design/laptop_lidar)
capture: ./data/reproduce/downloads/capture_0ee98f444e72/9a84cd46da

fit_lidar: measured hinge and opening over time
fit_lidar: 3D replay from the phone's camera

fit_lidar said:
reference frame 850: given
floor levelling adjusted the estimated camera pose by up to 10 mm
hinge: plate planes on 863/923 frames (415 closed and 395 half-open from plate_closed), 24 outliers dropped; axis 0.0 deg off the base plane; opening 108 deg on the reference frame, 0..117 deg over the video; plate spans 304 mm along the hinge
35 frames where the plate's mask was plate_closed's are measured from plate_closed: 0 shut (no open indicator: the plate track slid onto the shut plate's back, the designed fallback), 35 open where plate_closed's track followed the raised plate (measured from its points off the base), 0 open without a raised plate
model: base 14 mm thick; plate 309 x 205 mm, 3.8 mm thick; hinge -107..9 deg; replay covers lid 95%, base 94% of the SAM masks
Images: the hinge drawn on the reference frame with the opening over time, then the replayed model (colored) over the video at its worst frames and a few others.

physics said:
base: aluminum, 0.900 kg over 0.920 l of solid geometry (978 kg/m^3)
lid: aluminum, 0.500 kg over 0.269 l of solid geometry (1860 kg/m^3)
lid_joint: held by itself on 125 frames between -2 and 8 deg; gravity torque there up to 0.227 N m -> friction 0.283 N m. Released in MuJoCo for 2 s at 5 held angles it moves 0.1 deg (without that friction: 12 deg)

quantity                                           published      reproduced     difference
-------------------------------------------------  -------------  -------------  ----------
reference frame (the joint's zero)                 850            850            same
lid size, mm                                       308.9 x 205.2  308.9 x 205.2  +0.005 / +0.001
lid thickness, mm                                  3.84           3.84           0
base thickness, mm                                 13.62          13.62          0
hinge range (joint limits), deg                    -106.6 .. 9.4  -106.6 .. 9.4  +0.02 / 0
opening seen, deg (0 = shut)                       0.5 .. 117.4   0.5 .. 117.3   0 / -0.02
opening on the reference frame, deg                107.6          107.6          -0.02
frames with a measured lid plane                   863 of 923     863 of 923     0
  of them from plate_closed's mask: shut / raised  415 / 360      415 / 395      0 / +35
hinge fit residual, mm                             4.65           4.65           -0.005
lid span along the hinge, mm                       303.9          303.9          +0.001
hinge line vs the published one                                                  0.003 deg, 0.03 mm apart
opening on each frame both measured (863), deg                                   median 0.00, max 0.31
replay covers the lid's SAM masks, %               95             94.5           same to the whole %
replay covers the base's SAM masks, %              94             93.8           same to the whole %
hinge friction, N m                                0.284          0.283          -0.0001
released for 2 s with it, worst drift, deg         0.08           0.08           0
released without it, worst drift, deg              11.5           11.5           0

Reference frame pinned to the published fit's (model.json: 850). --reference auto lets today's code choose it; docs/REPRODUCE.md says why it takes another frame on this capture.
Published coverage is the provenance note's (whole percents); the masses are inputs from Spark's saved estimate: base 0.9 kg, lid 0.5 kg.
Wrote model.urdf, model.xml, meshes/, replay3d.mp4, hinge.jpg, reproduce.md and reproduce.json in ./data/reproduce/laptop_lidar (replay3d.mp4: the model drawn over the video from the phone's own camera poses).
Took downloads, unpacking and checks 81 s, fit, replay and physics 209 s, total 290 s (load average 67.0 at the end).
```

The difference column is reproduced minus published, at a finer precision than the values
shown.

## What each number means

| quantity | meaning |
|---|---|
| reference frame | the frame the fit measures from: the joint's zero is the lid's pose there, and the lid's inner face, the base's top and the floor are measured on it. By default pinned to the published fit's (850, from `model.json`); `--reference auto` lets today's code choose (below) |
| lid size | width along the hinge x depth away from it, from the lid's back on the shut frames (its depth points turned back about the hinge to the reference pose, 1st to 99th percentile) |
| lid thickness | how high the shut lid's back sits above the base's top (median over the shut frames) |
| base thickness | the base's top plane to the floor plane on the reference frame |
| hinge range (joint limits) | 1st and 99th percentile of the measured joint angle, written as the URDF/MJCF limits; 0 is the reference pose and negative closes (the sign follows the fitted axis) |
| opening seen | the same range as an opening angle, 0 = shut: it doesn't depend on the reference frame |
| frames with a measured lid plane | frames whose lid plane was fitted from depth; "shut" ones measure the lid from its closed back (plate_closed, "gray laptop lid"), "raised" ones from that track's points off the base while the lid lifts |
| hinge fit residual | RMS distance of the fitted hinge line from the lid's planes |
| hinge line vs the published one | angle between the two axes and how far apart the lines are (both in the reference camera's coordinates, so shown for the same reference frame only) |
| opening on each frame | the published and reproduced opening compared frame by frame |
| replay covers ... SAM masks | over the frames, the mean share of the part's SAM mask that the rendered model covers (the base on frames where the keyboard shows); the published value is the provenance note's in `model.json`, in whole percents |
| hinge friction | 1.25 x the largest gravity torque on the lid at the angles where it stood by itself (Spark's hand-free spans), leaving out where it lies on the base |
| released ... worst drift | how far the lid moves in 2 s when the model is let go in MuJoCo at five of those angles, with the measured friction and without it |

The masses are inputs, Spark's estimate: base 0.9 kg, lid 0.5 kg.

## Why some numbers differ slightly

The published model was fitted on 10/29 at 21:28 KST, with the code of commit `121cf40`. Run
through that commit's code, the same bundle gives the published `hinge.json` and `model3d.json`
bit for bit (checked: `git archive 121cf40` with the agent's `fit_lidar` tool on these tracks and
roles). Today's code measures the same video slightly differently in one place: since `29b00e4`
(10/29 22:49) the frames where SAM's screen track had slid onto the lid's back while the lid was
raised (35 of them) are measured from the lid's-back track instead, from its points off the base
(395 raised frames instead of 360). With that one step turned off, today's code gives the
published files bit for bit as well (checked the same way). The step moves the hinge line by
thousandths of a degree and hundredths of a millimetre, the lid's outline by 0.005 mm and the
opening on single frames by up to 0.3 deg (median 0.00), and the friction with them
(0.2835 -> 0.2834 N m). Coverage is published in whole percents.

## Today's own choice: `--reference auto`

Left to choose, today's code takes another reference frame, so every size is measured from
another view. Its rule (`Capture.ref_with`) takes the frame where the screen's mask is largest
among frames that also show the base well and the keyboard: since `5e20bef` (10/30) the
reference must show the open indicator, so that a shut lid can't be taken for an open one. The
keyboard's SAM track has no usable mask on frame 850 (nor on frame 0), the two frames it was
anchored on, so the rule takes frame 1, the next-largest view of the screen. From the same fresh
clone, run after the one above (4 min 36 s; the downloads reused):

```
$ .venv/bin/python scripts/reproduce.py --bundle laptop_lidar_tracks.zip --reference auto
...
fit_lidar said:
reference frame 1: where the plate is seen largest
floor levelling adjusted the estimated camera pose by up to 14 mm
hinge: plate planes on 863/923 frames (417 closed and 393 half-open from plate_closed), 24 outliers dropped; axis 0.0 deg off the base plane; opening 115 deg on the reference frame, 0..120 deg over the video; plate spans 300 mm along the hinge
35 frames where the plate's mask was plate_closed's are measured from plate_closed: 0 shut (no open indicator: the plate track slid onto the shut plate's back, the designed fallback), 35 open where plate_closed's track followed the raised plate (measured from its points off the base), 0 open without a raised plate
model: base 12 mm thick; plate 311 x 208 mm, 2.5 mm thick; hinge -114..2 deg; replay covers lid 95%, base 92% of the SAM masks
...
physics said:
base: aluminum, 0.900 kg over 0.766 l of solid geometry (1175 kg/m^3)
lid: aluminum, 0.500 kg over 0.176 l of solid geometry (2847 kg/m^3)
lid_joint: held by itself on 136 frames between -9 and 1 deg; gravity torque there up to 0.222 N m -> friction 0.277 N m. Released in MuJoCo for 2 s at 5 held angles it moves 0.1 deg (without that friction: 11 deg)

quantity                                                              published      reproduced     difference
--------------------------------------------------------------------  -------------  -------------  ----------
reference frame (the joint's zero)                                    850            1              other frame
lid size, mm                                                          308.9 x 205.2  311.1 x 208.3  +2.210 / +3.083
lid thickness, mm                                                     3.84           2.46           -1.381
base thickness, mm                                                    13.62          11.67          -1.945
hinge range (joint limits), deg (zero at each fit's reference frame)  -106.6 .. 9.4  -114.0 .. 2.2  -7.37 / -7.20
opening seen, deg (0 = shut)                                          0.5 .. 117.4   0.2 .. 119.9   -0.30 / +2.51
opening on the reference frame, deg                                   107.6          114.5          +6.88
frames with a measured lid plane                                      863 of 923     863 of 923     0
  of them from plate_closed's mask: shut / raised                     415 / 360      417 / 393      +2 / +33
hinge fit residual, mm                                                4.65           4.67           +0.015
lid span along the hinge, mm                                          303.9          299.8          -4.158
opening on each frame both measured (862), deg                                                      median 0.80, max 11.03
replay covers the lid's SAM masks, %                                  95             95.3           same to the whole %
replay covers the base's SAM masks, %                                 94             91.8           -2 pts
hinge friction, N m                                                   0.284          0.277          -0.0064
released for 2 s with it, worst drift, deg                            0.08           0.07           -0.002
released without it, worst drift, deg                                 11.5           10.9           -0.66
```

So the lid comes out 311 x 208 mm instead of 309 x 205 (the 13" MacBook Pro's lid is 304 x 212),
the base 2 mm and the lid 1.4 mm thinner, and the replay covers the base's masks 2 points less.
The joint limits move by the 7 deg between the two frames' openings, since the joint's zero is
the reference pose; the opening as such (0 = shut) agrees to 0.8 deg on the median frame and
within 1.3 deg on 90% of them; 4 frames differ by more than 5 deg, all in frames 232-252, where
both fits' angles jump from frame to frame. Both fits read the same saved tracks; they differ
only in the frame they measure from.

## How long it takes

On the MacBook (M1, 16 GB), from a fresh clone: `uv sync` 1 s (packages from uv's cache), then
4 min 51 s for the command above (`time`: 399 s user, 29 s system): 81 s to download the 39 MB
capture, unpack and check both files, 209 s for the fit, the replay and the physics. Other jobs
were running on the machine meanwhile (load average 365 when it started, 67 when it ended; it
has 8 cores), so an idle machine should be faster; an earlier fresh clone under a load average
of 130-367 took 9 min 22 s. The longest step is the replay: 923 frames rendered and scored at
1920 x 1440. A second run reuses the downloads and the unpacked capture.

## Machines

Tested on a MacBook (M1, 16 GB, macOS) with `uv sync` (Python 3.12, the versions in `uv.lock`).
Linux isn't tested: on a machine without a display MuJoCo's renderer needs `MUJOCO_GL=egl` (or
`osmesa`), and the replay is written as H.264 by OpenCV; the first step checks both and stops
with a message if either is missing.
