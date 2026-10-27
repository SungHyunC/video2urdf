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
1. Ingest: split the video into frames. Shoot on a tripod with the camera roughly facing the motion plane, and keep a credit card in view for metric scale.
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
- Home desktop (main machine, has a screen): coding, phone videos, viewers, demo recording.
- A100 server (headless Ubuntu):
  - Glimmer server
  - batch renders with `MUJOCO_GL=egl`
  - optional local SAM 3.1

  Reach it with `ssh -L 8000:localhost:8000 <host>`.
- Every model sits behind one OpenAI-compatible base URL. Switching between the Meta API and local open weights is a config change.

## Model notes (check against current docs)
- Muse Spark 1.3 (`muse-spark-1.3`): takes image, video and document input, supports tool calling, and works with the OpenAI SDK.
- SAM 3.1 API (`sam-3-1`): $0.20 per 1k video frames. Input limits and the response schema are undocumented, so test them first.
- SAM 3.1 open weights: on Hugging Face (reportedly `facebook/sam3.1`, gated, SAM license).
- Muse Glimmer (`meta-models/Muse-Glimmer-30B`):
  - Dense 29.6B with a vision encoder, Apache-2.0.
  - Needs ≥24 GB VRAM even at 4-bit, so it doesn't fit the 16 GB desktop.
  - No Ollama tag. Use vLLM, SGLang, llama.cpp or LM Studio.

## Schedule (KST)
| Date | Work |
|---|---|
| 10/27 | Repo skeleton, synthetic video generator, Glimmer server up |
| 10/28 | SAM 3.1 part tracking on 3 real videos; decide whether stickers are needed |
| 10/29–30 | Joint estimation, scale, URDF, MuJoCo replay overlay. **Checkpoint 10/30: the lamp moves correctly in sim** |
| 10/31–11/1 | Agent fix loop, Three.js viewer, 3 real objects |
| 11/2 | Robot-arm interaction scene **or** Check mode (pick one) |
| 11/3 | Get outside users to try it; README |
| 11/4 | Record demo, post on X, submit before 11/5 11:00 KST |
