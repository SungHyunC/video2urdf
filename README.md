# Video2URDF

Film an articulated object (desk lamp, scissors, laptop lid, drawer) with your phone and get a **measured joint model**
in URDF/MJCF. Video2URDF then proves the model by replaying it in MuJoCo on top of the video.

Built for the Meta Global AI Developer Hackathon (Oct 26 – Nov 4 2026) with Meta's SAM 3.1 and Muse models.

## Pipeline

| Stage | What it does | Status |
|---|---|---|
| Ingest | split the video into frames; a credit card in view gives metric scale | planned |
| Track | SAM 3.1 tracks parts from text prompts with stable IDs (fallback: colored stickers) | planned |
| Estimate | joint type, axis/center by least squares from relative 2D part motion; limits from the observed range | planned |
| Reason | Muse Spark / Muse Glimmer names parts, builds the kinematic tree, resolves conflicts | planned |
| Emit | URDF + MJCF | planned |
| Verify | MuJoCo replay overlaid on the video, silhouette IoU and joint-angle error, agent loop re-fits bad joints | planned |
| Check ("URDF Doctor") | flag sign flips and offsets in an existing URDF from a video or joint logs | planned |
| **Synthetic ground truth** | MuJoCo videos of mechanisms with known joints, masks and GT | **done** |

## Quickstart

Python 3.10+.

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows; on Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"
pytest                            # 36 tests, renders small videos
v2u synth --preset lamp           # -> data/synth/lamp_s000/
```

On a headless Linux server, render with EGL: `MUJOCO_GL=egl v2u synth ...`.

## Synthetic videos with ground truth

```bash
v2u synth --list                                  # cart_pendulum drawer lamp laptop pendulum random scissors
v2u synth --preset random --count 50 --tilt 8     # benchmark set: 1-3 random revolute/prismatic joints
v2u synth --preset scissors --noise 3             # hand-held drift of the root is on by default for scissors
```

Each folder holds:

| File | Contents |
|---|---|
| `video.mp4` | H.264 video, what the pipeline sees (default 1280x720, 30 fps, 5 s) |
| `masks.npz` | `labels` (T, H, W) uint8: 0 background, i = i-th link, 255 = credit card |
| `gt.json` | camera (K, R, t), plane-to-image homography, card corners, model, per-frame joint values, anchors and link poses |
| `scene.xml` | MJCF that reproduces the render |
| `overlay.png` | first frame with GT pivots, prismatic axes and card corners drawn on top |

Conventions:
- World x right, z up, y away from the camera. Motion happens in the x-z plane.
- Revolute joints turn about -y, so a positive angle is counter-clockwise in the image.
- The camera uses OpenCV conventions, with pixel centers at integers and a principal point of ((W-1)/2, (H-1)/2).
- A joint's anchor is its joint-frame origin at q = 0, fixed to the parent: the pivot of a revolute joint, the slide origin of a prismatic one.
- Every joint visits both of its limits, so the observed range equals the GT limits.
- The ISO/IEC 7810 ID-1 card (85.60 x 53.98 mm) stands in the motion plane, so its four corners give the metric homography.

## Models

Every model sits behind one OpenAI-compatible base URL. Switching between the Meta Model API and local open
weights is one line in `.env` (copy `.env.example`):

```bash
V2U_PROFILE=meta     # https://api.meta.ai/v1, MODEL_API_KEY, muse-spark-1.3
V2U_PROFILE=local    # vLLM on localhost:8000, muse-glimmer (open weights)
v2u llm-check                                         # text round trip
v2u llm-check --image data/synth/lamp_s000/overlay.png   # vision input
```

### Open-weights path (Muse Glimmer, Apache-2.0)

On a GPU server (A100 80 GB serves bf16; on 40 GB the script falls back to the FP8 build):

```bash
pip install "vllm>=0.28"
bash scripts/serve_glimmer.sh      # serves meta-models/Muse-Glimmer-30B as "muse-glimmer" on 127.0.0.1:8000
```

On the desktop:

```bash
ssh -L 8000:localhost:8000 <gpu-host>
# .env: V2U_PROFILE=local
v2u llm-check --image data/synth/lamp_s000/overlay.png
```

## Rights and licenses

All code in this repository was written during the hackathon (Oct 26 – Nov 4 2026); no pre-existing code is used.
Synthetic media is rendered by this code. Dependencies keep their own licenses: MuJoCo (Apache-2.0), OpenCV
(Apache-2.0), NumPy (BSD-3), imageio and imageio-ffmpeg (BSD-2), OpenAI Python SDK (Apache-2.0), python-dotenv (BSD-3).
Muse Glimmer weights are Apache-2.0; SAM 3.1 weights are under the SAM License.

Code: [MIT](LICENSE).
