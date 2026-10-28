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
| Meta Model API | $150 | Main path: tracking, reasoning, agent loop | `sam-3.1`; `muse-spark-1.3-contributor` for dev, `muse-spark-1.3` | `https://api.meta.ai/v1`, Responses or Chat Completions for Spark, Responses only for SAM; key `MODEL_API_KEY` |
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
- Known gaps: link geometry is one box from the reference-frame mask, too thick when the camera sees a face obliquely (laptop lid). Replay IoU is harsh on thin parts. Spark tends to accept low worst-frame IoU as "thin part" without checking.
- Behind schedule: the Glimmer endpoint.
- Next, in order:
  1. Three.js viewer with the video overlay.
  2. Robot-arm scene (11/2 pick): a MuJoCo arm operating a captured object.
  3. Open-weight path: run the agent on Muse Spark Open (Baseten: AGENT_BASE_URL/AGENT_MODEL/AGENT_API_KEY_ENV) and SAM 3.1 weights on Modal.
  4. Reshoot the drawer from the side.
