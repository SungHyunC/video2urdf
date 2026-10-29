# Video2URDF

Video2URDF turns a phone video of an articulated object (a desk lamp, scissors, a laptop lid, a drawer) into a measured joint model in URDF/MJCF. It then proves the model by replaying it in MuJoCo against the video.

- User: a robotics developer who needs a sim model of a real object.
- Job: one short video in, a URDF that moves like the real thing out.

Built solo for the Meta Global AI Developer Hackathon (Cerebral Valley, Indie track, Oct 26 – Nov 4 2026).

## Hackathon rules that constrain every change
- Deadline: Nov 4 2026 9:00 PM ET = **Nov 5 11:00 KST**. Submit a public repo and a 3-minute narrated demo video (no slides).
- Only work done Oct 26 – Nov 4 counts. Don't bring in pre-existing code. If any is used anyway, the video must say so up front.
- The project must use Meta models: Muse Spark or SAM 3.1 through the Model API, or open weights (Muse Glimmer, Muse Spark Open).
- Banned project types:
  - medical or mental-health advice
  - basic RAG
  - simple image analyzers
  - **Streamlit apps and dashboard-centric apps**
  - education chatbots
  - job screeners
  - nutrition coaches
  - personality analyzers
  - sports analyzers or coaches
  - political or war topics

  Because of this, the UI is a Three.js URDF viewer with a video overlay, never Streamlit.
- Use only code, data and media we have rights to, and follow open-source licenses. CC-BY media needs attribution. The participant must own all rights to the submission.
- Never commit API keys. Keep them in `.env`, which is git-ignored.
- Organizer answers (Discord, 10/28):
  - Every model the project calls at runtime must be a Meta model.
  - Hackathon credits cover only `-contributor` models; `muse-spark-1.3` and other opt-out variants bill the card. Always use `muse-spark-1.3-contributor`. Contributor inputs may train Meta models and zero retention isn't available. The model list has no `sam-3.1` contributor variant, so check on dev.meta.ai whether SAM usage hits credits or the card.
  - SAM 3.1 takes images, videos and frame batches and keeps object identity across frames. Prompts are nouns ("lamp arm"), not sentences that need reasoning. Docs: dev.meta.ai/docs/media-segmentation.
  - Open Weight bonus: Glimmer or Spark Open on our own hardware or partner inference, verified by the partner. How much of the pipeline must be open weights is unanswered, so keep a mode that runs the whole pipeline on open weights only.
  - Submissions can be edited until the deadline, not after; submitting early gains nothing. The demo is cut at 3:00; sped-up footage is fine. Track and bonus prizes can stack.
  - Most Viral counts likes + reposts + comments on the one X post linked at submission, until 11/8. Posting early helps; the user decides when.
  - Judges read the commit history to check the work is new: commit in small steps. The README must list the open-source models and repos used and what we built on top.
  - Claude Code is allowed and needn't be disclosed.
- Competition context (10/28): no other team does URDF or articulated objects. Closest are CCTV accident reconstruction (physics from video) and SnapPart (dimensioned 3D from photos). One product-video team pitches a self-checking fix loop, so make our MuJoCo replay and numeric verification obvious.
- Judging weights:
  - Impact 30
  - Meta tooling 25 ("beyond basic integration")
  - Demo 25
  - Technical depth 20
- Bonus prizes we're aiming for:
  - Open Weight ($10K): partners check that it really runs on open weights, so the README needs exact commands for the local path.
  - Agentic ($5K).
  - Most Viral ($20K): judged on engagement with an X post.

## Pipeline
1. Ingest: split the video into frames. Shoot on a tripod with the camera roughly facing the motion plane. Keep a card-sized reference (ID-1, 85.6 × 54 mm) in view for metric scale. Use a loyalty card or similar, never a real payment card, because frames go to the Meta API and into the public demo.
2. Track: SAM 3.1 tracks parts from text prompts and keeps their IDs stable. Fallback: colored sticker markers ("blue sticker").
3. Estimate: from the relative 2D motion of each part pair, find the joint type (revolute or prismatic) and the axis or center by least squares. Take joint limits from the observed range.
4. Reason: Muse Spark (or Glimmer) names the parts, builds the kinematic tree and resolves conflicts.
5. Emit: write URDF and MJCF.
6. Verify: replay the model in MuJoCo and overlay it on the video. Score silhouette IoU and joint-angle error per frame. An agent loop re-fits joints that score low.
7. Check mode ("URDF Doctor"): given an existing URDF plus a video or joint logs, flag sign flips and offsets. Demo it on LeRobot SO-101 data and the public SO-101 URDF with bugs injected on purpose.

- v1 scope: planar mechanisms with 1–3 joints. Full 3D and real meshes are stretch goals.
- Ground truth: synthetic MuJoCo videos with known joints. Report axis error (mm) and angle error (deg).
- Positioning: Articulate-Anything retrieves meshes from a dataset, and Artifex generates models from text. We measure real motion and check existing models.

## Machines
- **The MacBook (M1, 16 GB) is the main machine** (decided 10/28). Coding, phone videos, MuJoCo, viewers and demo recording all happen here, and Claude Code runs here.
  - System Python is 3.8, which is too old. Use uv with Python 3.12 in `.venv/`.
  - MuJoCo runs natively on Apple Silicon. Launch the interactive viewer with `mjpython`, not `python`. Don't set `MUJOCO_GL=egl`, which is Linux only.
- No local GPU is needed. All heavy compute runs on hackathon credits (next section). Modal replaces the A100 server, and the home desktop is optional.

## Backends and credits
Keys live in `.env` (template: `.env.example`). All four credits arrived on 10/28.

| Backend | Credit | Role | Models | How to call |
|---|---|---|---|---|
| Meta Model API | $150 | Main path: tracking, reasoning, agent loop | `sam-3.1`; `muse-spark-1.3-contributor` only (credits cover only `-contributor`) | `https://api.meta.ai/v1`, Responses or Chat Completions for Spark, Responses only for SAM; key `MODEL_API_KEY` |
| Baseten | $100, 2M TPM | Open-weights reasoning for the Open Weight bonus | `meta-models/muse-spark-open` (the listed ID; `-1026` returns 404) | `https://inference.baseten.co/v1` Chat Completions, key `BASETEN_API_KEY` |
| Modal | $100 | GPUs: Muse Glimmer endpoint, SAM 3.1 open weights, batch renders | Muse Glimmer 30B (Dedicated Endpoint), `facebook/sam3.1` | `modal setup` (token in `~/.modal.toml`), then `modal endpoint create --model <id>`; HTTP calls need a proxy token (`MODAL_PROXY_TOKEN_ID`, `MODAL_PROXY_TOKEN_SECRET`) |
| Fireworks | $101 | Spare Muse Spark Open. Its credits never expire, so use it last | `accounts/fireworks/models/muse-spark-open` | `https://api.fireworks.ai/inference/v1` Chat Completions, key `FIREWORKS_API_KEY` |

- Every model sits behind an OpenAI-compatible API, so switching between the Meta API and open weights is a config change. Meta Spark accepts both Responses and Chat Completions (verified 10/28), so Chat Completions works everywhere except SAM. Modal Endpoints accept both.
- While Meta free credits are in use, inputs may be used for product improvement, so the contributor tier costs no extra privacy. Never film anything sensitive.
- Modal Dedicated Endpoints bill while they're up (an A100 80 GB is about $2.50/hour). Use scale-to-zero and delete endpoints when done.
- `scripts/smoke_test.py` makes one cheap call per backend. All seven checks passed on 10/28.
- Spark reasons before answering, so keep `max_output_tokens` ≥ 1024 or the text comes back empty.
- Not yet verified:
  - the exact Modal model ID for Glimmer

## Model notes (check against current docs)
- Muse Spark 1.3: takes image, video and document input, and supports tool calling.
- Muse Spark Open: the open-weights Spark. It takes vision and audio input and always reasons.
- SAM 3.1 API (`sam-3.1`, Responses only): $0.20 per 1k video frames, min 50 frames per video request, MP4/MOV H.264/H.265, ≤15k frames, ≤16 objects. Code in `video2urdf/sam.py`; docs at dev.meta.ai/docs/sam/*.
  - Output is text, one line per frame: `<Nf>id<|box;x1;y1;x2;y2;w;h;c|><|mask;x=0;y=0;data=H,W,payload|>`. The mask is a crop anchored at the box's (x1, y1). Decode with `meta-sam-parser` (`decode_mask_to_raster`).
  - One concept per request. Concept prompts on video only start tracking where the concept is first confident, then go forward, so parts get missed early. Instead, find each part with a concept prompt on a few key frames (image calls), then track all parts in one box-prompt video request with `propagation_direction=both` (`scripts/track_video.py`).
  - Prompt wording matters: "blue lamp arm" and "green drawer" found nothing on synthetic videos, while "blue rod" and "green box" worked. A prompt may cover only part of a rigid body ("blue rod" skips the lamp head), which is fine for joint estimation.
- SAM 3.1 open weights: `facebook/sam3.1` on Hugging Face (`sam3.1_multiplex.pt`, gated, SAM license). Access is granted.
- Muse Glimmer (`meta-models/Muse-Glimmer-30B`):
  - Dense 29.6B with a vision encoder, Apache-2.0.
  - Needs ≥24 GB VRAM even at 4-bit, so it runs on Modal, not locally.
  - For the README's local path: vLLM, SGLang, llama.cpp or LM Studio. There's no Ollama tag.

## Schedule (KST)
| Date | Work |
|---|---|
| 10/27 | Repo skeleton, synthetic video generator, Glimmer endpoint up |
| 10/28 | SAM 3.1 part tracking on 3 real videos; decide whether stickers are needed |
| 10/29–30 | Joint estimation, scale, URDF, MuJoCo replay overlay. **Checkpoint 10/30: the lamp moves correctly in sim** |
| 10/31–11/1 | Agent fix loop, Three.js viewer, 3 real objects |
| 11/2 | Robot-arm interaction scene **or** Check mode (pick one) |
| 11/3 | Get outside users to try it; README |
| 11/4 | Record demo, post on X, submit before 11/5 11:00 KST |

## Status (10/28)
- Done:
  - Mac environment: uv, Python 3.12 `.venv`, `pyproject.toml` + `uv.lock` (`uv sync` rebuilds it).
  - `scripts/smoke_test.py`: all four backends verified, including the Fireworks model ID `accounts/fireworks/models/muse-spark-open`.
  - `modal setup`: token in `~/.modal.toml`, workspace `honeybeelawn`.
  - SAM 3.1 tracking pipeline (`video2urdf/sam.py`, `scripts/track_video.py`) checked on synthetic videos. Every frame tracked; IoU vs ground truth: red rod 0.955, drawer 0.984.
  - Synthetic generator (`video2urdf/synth.py`, `scripts/make_synth.py`): presets `lamp` (2 revolute) and `drawer` (1 prismatic). Writes mp4, MJCF and `gt.json` (joint values and pixel anchors per frame). `scripts/eval_tracks.py` scores tracks against MuJoCo silhouettes.
  - Real videos (`data/real/`, git-ignored): `stand.mov` ("base plate", "vertical bar", "gray panel") and `notebook.mov` ("laptop keyboard", "laptop lid") are tracked on every frame, through hand occlusion. **No stickers needed.** Check with `open data/tracks/<name>/overlay.mp4`.
  - Not sent to the API: `drawer.MOV` (faces of the user and a bystander), `card.MOV` (looks like a real payment card), `notebook_card.MOV` (oblique, variable 11 fps).
  - The stand camera shifts between frames (the base moves in the image), so joint estimation must use motion relative to the base, not image coordinates.
  - Joint estimation → URDF/MJCF → MuJoCo replay (`scripts/fit_model.py`; `video2urdf/motion.py`, `joints.py`, `emit.py`, `replay.py`). Writes `model.urdf`, `model.xml`, `joints.json` and `replay.mp4` into the track folder.
    - Part poses: ECC registration of each part's mask to its reference-frame mask. Frames where the mask is occluded (registered IoU < 0.9) use KLT points tracked on the part's own surface, re-anchored to the nearest clean frame. The background can't stand in for camera motion: a handheld phone gives parallax between the wall and the object.
    - Joints: least-squares hinge center from relative poses, with outlier frames dropped; prismatic if rotation range < 5°. Tree: Prim's from the first prompt, with cost = residual + hinge distance from both parts' boxes.
    - Synthetic lamp: correct tree and types, hinge position error 1.7 / 2.5 mm, angle error 0.08° / 0.38° mean. Synthetic drawer: prismatic, axis direction error 0.00°, position error 3.2 mm (the drawer face is nearer the camera than the y = 0 scale plane).
    - Real notebook: one hinge, 0–90°, residual 4.3 mm, replay IoU ~0.6. Real stand: base → bar → panel, bar −51–7°, panel joint barely used (±5°), replay IoU 0.78 / 0.46 / 0.68. Scale is assumed (lid 0.212 m, bar 0.30 m) because there was no card.
    - SAM mask decoding is pure Python (~0.3 s per large mask). `load_tracks` decodes in parallel once and caches `tracks.sam.npz`.
  - **Reason step / agent** (`video2urdf/agent.py`, `scripts/run_agent.py`): Muse Spark (`muse-spark-1.3-contributor`, chat completions with tools and images) drives the run. Tools: look, segment (SAM image call, annotated image back), measure_scale (ID-1 card), track, fit (tree/fixed overrides, worst replay frames back), finish. 20-turn budget, forced finish. Log with every image Spark saw: `<track>/agent/log.md`, report in `agent/report.md`.
    - stand_2: base → lower_arm (0–30°) → upper_arm (−81–0°), residuals 1.5 / 2.4 mm, card scale 5094 px/m.
    - notebook_2: Spark saw its first fit was wrong (base tracked 319/408, 25° range), re-segmented on other frames and re-tracked: 408/408, 0–77°, residual 1.7 mm.
    - drawer_2: fails. Shot from above at an angle, so the drawers slide toward the camera and are hidden when closed (not planar). Reshoot from the side.
  - SAM API findings: `propagation_direction=both` silently drops the backward pass (it fails with server_error on phone video), so `track_boxes` tracks forward, then forward through an ffmpeg-reversed clip of the frames before the prompt, deduped per (frame, object). Forward output starts at the earliest prompt frame for every object.
  - Speed: one_bit masks decode with a numba port of the reference decoder (`video2urdf/fast_mask.py`, ~74x, verified identical by `scripts/check_fast_mask.py`; derived from meta-sam-parser under the SAM License, copy in `LICENSES/SAM_LICENSE`). Comparing a numpy array to a MuJoCo enum is elementwise in Python; cast with int(). A 620-frame fit takes ~1 min.
  - Frame counts come from decoding (`pipeline.video_info`): the header count is off by one on variable-frame-rate video.
  - **Robot demo** (`scripts/robot_demo.py`, `scripts/make_reel.py`): a Franka Panda (MuJoCo Menagerie, Apache-2.0, fetched by `scripts/fetch_assets.sh` into git-ignored `third_party/`) closes the notebook_2 laptop in MuJoCo physics by pushing along the measured hinge (IK + contact). The laptop's width (30 cm) is assumed: a side view doesn't measure it. Reel: `data/tracks/notebook_2/reel.mp4` (phone video | replay | robot).
    - MuJoCo skips parent-child contacts except when the parent is welded to the world; the demo excludes them explicitly.
  - **3D experiment, stage A (oracle depth) passed** (`video2urdf/lift3d.py`, `scripts/fit3d.py`, `synth.render3d`): points tracked in each part's mask, lifted with per-frame depth, RANSAC Kabsch per part, hinge refit on point motion, parents down the chain use their fitted motion. Camera poses cancel out of parent-child motion, so only depth is needed. Oblique (az 40°, el 25°) lamp with handheld wobble: shoulder axis 0.5–0.85° / 4 mm / 0.1°, elbow 1.4–1.6° / 8 mm / 0.9°. Drawer sliding toward the camera (the drawer_2 failure case): prismatic, axis 0.23°, travel error 0.6 mm.
    - Mask-edge pixels read background depth; drop them with `depth_edges` before any 3D fit.
  - **Geometry (A)**: each link is its profile voted over the whole video (masks moved back to the reference frame by the part's pose), extruded to an STL mesh with exact inertia, in the part's median colour (`video2urdf/shape.py`). Width across the plane and, for oblique views, thickness can be set (`--widths`, `--thickness`, or by the agent). Synthetic lamp replay IoU 0.95/0.87/0.77 -> 0.96/0.91/0.87.
  - **3D stage B, iPhone LiDAR** (Stray Scanner: `video2urdf/stray.py`, `scripts/fit_lidar_hinge.py`): laptop filmed from above at an angle, no card. ARKit odometry gives the static scene's motion (convention checked by aligning depth clouds: camera-to-world, no axis flip); the lid is tracked by fitting its plane in each frame's depth (works while it shows the screen, then its aluminium back). Hinge drawn exactly on the real hinge, lies in the keyboard plane (2 deg), lid spans 297 mm along it (13" MacBook Pro display ~300 mm), opening 102 deg on the reference frame, closed reads -10 deg (bias), reopening tracked.
  - SAM API: mask rasters are capped at 1024 px on the long side; larger boxes come back downscaled (fixed in `Detection.full_mask`; this also shortened stand_2's tall bars). One object can take box anchors on several frames; `track_video.py` anchors every key frame (8) where the part is found, which re-acquires parts that disappear and return. Dropped stream connections are retried.
  - **3D model from LiDAR** (`scripts/build_lidar_model.py`, `video2urdf/emit3d.py`, `replay.replay3d`): model frame z up off the base, x along the hinge. Measured from the capture: base 12 mm thick (top surface to floor), lid outline 302 x 215 mm from its closed aluminium back turned back about the hinge (13" MacBook Pro lid: 304 x 212), lid thickness from how high its back sits when closed (clipped at the 3 mm floor, so not really measured), hinge range from the tracked angle. Replay renders the model from the iPhone's own camera pose every frame: it stays on the laptop through the moving camera; SAM masks covered 92% (base), 79% (lid).
  - Robot demo now places any model from its geometry (turned so closing pushes away from the robot, hinge 0.5 m out, closed end = the limit where the pushed link sits lowest). LiDAR laptop: closed 109.5 of 112 deg. Reels: `data/tracks/laptop_lidar/reel.mp4` (LiDAR), `data/tracks/notebook_2/reel.mp4` (2D).
  - **Agent drives the LiDAR path** (`video2urdf/lidar.py` holds the hinge fit and model build; `fit_lidar` tool; `run_agent.py` takes a Stray Scanner folder). Spark picks prompts and part roles (floor, base, plate, plate_closed, open_indicator), gives disappearing parts candidates on several frames, then fits. From scratch (`data/tracks_agent/laptop_lidar`): lid 304 x 215 mm, base 12 mm, hinge 0.6 deg off the base plane, opening -5..113 deg; it flagged the clipped lid thickness itself. Robot closes it 111 of 111 deg. Reel: `data/tracks_agent/laptop_lidar/reel.mp4`.
  - Robot demo: IK steps and joint targets are rate-limited; an IK branch jump used to whip the arm and leave the lid half closed.
  - The interactive MuJoCo viewer can't open from Claude Code's shell; launch it through Terminal (`osascript ... do script`) or have the user run `.venv/bin/mjpython -m mujoco.viewer --mjcf=<model.xml>`.
  - **Web viewer** (`docs/index.html`, data from `scripts/export_viewer.py` into `docs/models/`): static page, three.js from a CDN, no build. Phone video / model-over-video / robot demo on the left; the measured model in 3D on the right, its joints driven by the video's time; joint sliders; URDF/MJCF downloads; the agent's report. Serve with `.venv/bin/python -m http.server 8931 --directory docs` (port 8765 is taken by another local service). For GitHub Pages: serve `docs/` once the repo is public (ask first).
  - **Accuracy pass (10/28)**:
    - stand_2 refit after the 1024 px mask fix, with meshes: replay IoU base 0.86 -> 0.92, lower arm 0.63 -> 0.71, upper arm 0.62 -> 0.66; fit residuals 1.5 -> 1.0 mm and 2.4 -> 0.9 mm.
    - LiDAR: ARKit drifts ~9-14 mm in height over the video. `Capture.level_with(floor)` fits the floor plane per frame, drops implausible fits (> 4 deg), smooths over 31 frames, and removes the tilt and height. Closed lid now reads 1 deg (was -5), lid thickness is measured (2.6 mm; it used to read 9 mm *below* the base and was clipped).
    - Half-open frames without the screen mask now use the whole-object mask minus the base's top surface: plate measured on 697/923 frames (was 486). Lid replay coverage 66% -> 71%; base 89% -> 86%.
  - Robot demo: a lid opened past vertical leans back over the robot and can only be pushed from underneath, which drove the arm into the pedestal; the demo now starts the lid at most 80 deg open (inside its measured range) and the pedestal is 22 cm. All three models close to within 2-3 deg of their target.
- **Slide joints in 3D** (`lidar.fit_and_build_slide`, `fit_lidar joint="slide"`): translation-only ICP on the drawer's depth. Synthetic drawer toward the camera: axis 0.24 deg, travel error 0.17 mm. No real LiDAR drawer yet.
- **RBO benchmark** (TU Berlin RBO dataset, CC BY 4.0, Zenodo 1036660; `scripts/rbo_download.py`, `rbo_eval.py`, `rbo_bench.py`; raw data git-ignored in `data/rbo/`). Kinect RGB-D with motion capture; the agent runs each sequence from scratch, then it is scored.
  - Truth = the plate's motion relative to the base from the tf rigid-body frames (rb_poses_array orientations are unreliable). Operators push the whole object around (laptop04: 121 mm, 47 deg), so `Capture.follow(base)` tracks the base with KLT points lifted by depth (`part_poses_3d`); depth-only ICP slid along flat bases.
  - Hinge axis = median crossing of plate and base planes. Base top = `lidar.base_top`: the base face parallel to the floor; if none is within 30 deg, the floor normal (book05 was 89 deg off, now 2.5).
  - Results (10/29, `data/rbo/results/summary.md`, angle error mod 360): median of 13: axis 2.5 deg, position 47 mm, angle 3.3 deg. Laptops 0.9-2.2 deg / 12-31 mm / 1.9-3.3 deg; book03/05 1.3-1.4 deg; book13 16 deg; boxes mostly fail (SAM API loses lids and even the box on most 640x480 frames, re-anchoring doesn't fix it); treasurebox03 0.9 deg.
  - Masks under 0.05% of the frame count as lost (`Detection.usable`); SAM keeps IDs alive on 3-pixel masks after losing a part. The agent's track report says how many frames are usable. This let the closed/half-open fallback measure book frames (book05 16.9 -> 1.4 deg).
  - build_model picks closed frames by the reference opening, not q's sign (the axis sign flips q); LiDAR laptop now 312 x 207 mm, lid 1.7 mm, replay lid 96% / base 96%.
- **Open weights**:
  - `run_agent.py --open-weights`: agent on Muse Spark Open (Baseten), SAM via `SAM_BACKEND=open` (`sam.sam_client()`); `--sam api` keeps the Meta API segmenter. Baseten allows 8 images per request (`AGENT_MAX_IMAGES`, older images dropped).
  - Muse Spark Open ran the LiDAR laptop end to end in 9 turns (`data/tracks_open/laptop_lidar`): lid 312 x 208 mm, base 13 mm, 0..117 deg, hinge on the real hinge.
  - SAM open weights on Modal (`modal_apps/sam3_open.py`, deployed, A10G, scale-to-zero 120 s): image concept prompts on SAM 3.1 (`facebook/sam3.1`); box tracking on SAM 3's SAM2-style tracker (`facebook/sam3`), because the SAM 3.1 multiplex predictor only returns point/box objects on their prompted frames. Synthetic lamp: 120/120 frames, IoU vs hosted 0.970 / 0.952. Gotchas: sam3's mp4 reader needs decord (we pass a JPEG folder); a prompted frame only outputs the objects prompted on it (filled from the neighbour frame); `modal deploy` leaves warm old containers serving (stop them with `modal container stop -y`). ~3 frames/s on 1920x1440.
  - Full open-weights run (Spark Open + SAM on Modal), LiDAR laptop, 18 turns (`data/tracks_open_full`): lid 305 x 209 mm, 3..122 deg, lid replay 91%; base segmented in fragments, so base thickness wrong (-73 mm), flagged by Spark.
- Agent tool errors now log the full traceback in `agent/log.md`.
- Known gaps: Replay IoU is harsh on thin parts and understated when thickness is overridden. Spark tends to accept low worst-frame IoU as "thin part" without checking. fit_lidar handles one hinged plate only. Floor levelling can't see drift along the floor. Box lids and book covers track poorly on RBO (see above).
- Behind schedule: the Glimmer endpoint.
- Next, in order:
  2. Box lids: SAM loses them at 640x480 (tried: tiny-mask handling, re-anchoring, top-band closed planes; a top-plane-orientation closed test hurt laptops and was reverted).
  3. Multi-joint 3D: RBO pliers/foldingrule are too small for SAM at 640x480 ("black plastic part" fragments, "yellow ruler" as one piece); needs our own LiDAR capture (e.g. a desk lamp). Real drawer LiDAR capture too.
  4. README with exact commands, credits (RBO CC BY, Menagerie Apache-2.0, sam3 and meta-sam-parser SAM License, three.js MIT, mapbox-earcut ISC) and the benchmark table; publish the viewer on GitHub Pages (ask first).
  5. URDF Doctor on SO-101 / LeRobot data (needs camera calibration; not started).
- **URDF Doctor** (10/29, `video2urdf/doctor.py`, `scripts/urdf_doctor.py`, `examples/doctor/`): registers a foreign URDF's base to the measured base (PCA x4 + ICP, then refined on base + child at 7 openings), finds the URDF q per opening by 1-D search, reports axis direction/offset, sign, zero, limits vs the video, shape mismatch, and joint-log sign flip/offset. Laptop: spec URDF passes (2.3 deg / 8 mm); injected bugs found (tilt 6 -> 7.6 deg, hinge 25 -> 18 mm, limit 90 vs needed 127; log flip -1.00x; log offset 12 -> 11.3 deg). In-plane axis turn isn't resolvable (measured base outline is rough): a 6 deg in-plane turn read as 0.1 deg, so the example tilts out of plane. Mesh geoms are sampled over their triangles (vertices only outline flat parts).
- Repo public, GitHub Pages live (10/29): https://sunghyunc.github.io/video2urdf/ (master /docs, .nojekyll). Pushing needs `git -c http.postBuffer=524288000 push` (videos).
- SAM free path: SAM_BACKEND=open (Modal credits) instead of the Meta API sam-3.1 (billing to credits vs card unconfirmed).
- Viewer (10/29): laptop_lidar rebuilt with the current fit from data/tracks_agent (roles.json written from the agent's log): 309 x 205 mm, lid 3.8 mm, replay lid 95% / base 94%; robot closes to -77.8 of -77.1 deg.
