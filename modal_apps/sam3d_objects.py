"""SAM 3D Objects (facebook/sam-3d-objects, SAM License) on a Modal GPU: one frame and a part's
mask in, a coloured mesh of that part out, placed in the camera frame in metres.

Given our LiDAR depth as the pointmap, SAM 3D lays the object out in that metric frame instead
of guessing depth with its own monocular model (MoGe), so the generated mesh lands where the
measured part is. It also completes the sides the camera never saw.

Setup (once): request access on huggingface.co/facebook/sam-3d-objects, then
    modal secret create huggingface HF_TOKEN=hf_...      (already there for sam3_open.py)
    modal deploy modal_apps/sam3d_objects.py
    modal run modal_apps/sam3d_objects.py::download     (12 GB of checkpoints into a volume)
Client: scripts/sam3d_meshes.py.

GPU: the repo asks for >= 32 GB; an L40S (48 GB) fits. Its environment is CUDA 12.1 /
torch 2.5.1 with pytorch3d, kaolin, gsplat, spconv, xformers and flash-attn built for it.
"""

import io
from pathlib import Path

import modal

APP = "video2urdf-sam3d-objects"
SAM3D_COMMIT = "f91db411c50efee93d8db7aeb323885650f6f722"  # facebookresearch/sam-3d-objects
REPO = Path("/opt/sam3d")
WEIGHTS = Path("/weights")
CKPT = WEIGHTS / "checkpoints" / "hf"
TORCH_INDEX = "https://download.pytorch.org/whl/cu121"
KAOLIN_LINKS = "https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
FLASH_ATTN = ("https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/"
              "flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl")
# Pinned by the repo but never imported for inference, and some don't install on a fresh
# image (auto_gptq compiles CUDA kernels, nvidia-pyindex and pip-system-certs patch pip).
SKIP = ("auto_gptq", "nvidia-pyindex", "pip-system-certs", "sagemaker", "wandb", "bpy", "jupyter", "conda-pack",
        "mosaicml-streaming", "hdfs", "pymongo", "pdoc3", "black", "flake8", "autoflake", "usort", "pytest",
        "bitsandbytes", "optimum", "sentence-transformers", "hydra-submitit-launcher", "polyscope")

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0", "libgomp1", "ninja-build", "build-essential", "libegl1", "libxrender1")
    .pip_install("torch==2.5.1", "torchvision==0.20.1", "torchaudio==2.5.1", index_url=TORCH_INDEX)
    .run_commands(f"git clone https://github.com/facebookresearch/sam-3d-objects.git {REPO} "
                  f"&& cd {REPO} && git checkout {SAM3D_COMMIT}")
    .run_commands(
        # requirements.txt minus SKIP; torchaudio is already in, from the torch index
        f"cd {REPO} && grep -v -i -E '^({'|'.join(SKIP)})(=|@| |$)|^torchaudio' requirements.txt > /tmp/req.txt "
        f"&& pip install -r /tmp/req.txt --extra-index-url {TORCH_INDEX} --extra-index-url https://pypi.ngc.nvidia.com")
    .env({"FORCE_CUDA": "1", "TORCH_CUDA_ARCH_LIST": "8.0;8.6;8.9", "MAX_JOBS": "8"})
    .pip_install(FLASH_ATTN)  # first: requirements.p3d.txt pins flash_attn, which would build from source
    .run_commands(f"cd {REPO} && pip install --no-build-isolation -r requirements.p3d.txt")
    .run_commands(f"cd {REPO} && pip install --no-build-isolation -r requirements.inference.txt "
                  f"--find-links {KAOLIN_LINKS} --extra-index-url {TORCH_INDEX}")
    .run_commands(f"cd {REPO} && pip install --no-deps -e . && python patching/hydra")
    .pip_install("huggingface_hub[cli]<1.0", "trimesh==4.5.3")
    .env({"HF_HUB_DISABLE_TELEMETRY": "1", "PYTHONPATH": f"{REPO}/notebook"})
)
volume = modal.Volume.from_name("sam3d-weights", create_if_missing=True)
app = modal.App(APP, image=image)


@app.function(volumes={WEIGHTS: volume}, secrets=[modal.Secret.from_name("huggingface")], timeout=3600)
def download():
    """Checkpoints into the volume, laid out as the repo's doc/setup.md says (checkpoints/hf)."""
    import os
    import shutil
    import subprocess

    if (CKPT / "pipeline.yaml").exists():
        return "already there"
    tmp = WEIGHTS / "hf-download"
    subprocess.run(["huggingface-cli", "download", "facebook/sam-3d-objects", "--repo-type", "model",
                    "--local-dir", str(tmp), "--include", "checkpoints/*", "--token", os.environ["HF_TOKEN"]], check=True)
    CKPT.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(tmp / "checkpoints"), str(CKPT))
    shutil.rmtree(tmp, ignore_errors=True)
    volume.commit()
    return sorted(p.name for p in CKPT.iterdir())


@app.function(gpu="L40S", timeout=1200)
def check_imports() -> str:
    """The environment imports (before the weights arrive)."""
    import torch

    import sam3d_objects  # noqa: F401
    from inference import Inference  # noqa: F401  (notebook/inference.py)
    import pytorch3d, kaolin, gsplat, spconv, xformers, flash_attn  # noqa: E401,F401

    return f"torch {torch.__version__}, cuda {torch.cuda.is_available()} {torch.cuda.get_device_name(0)}"


def _p3d(points):
    """OpenCV camera (x right, y down, z forward) <-> PyTorch3D camera (x left, y up): its own inverse."""
    return points * points.new_tensor([-1.0, -1.0, 1.0]) if hasattr(points, "new_tensor") else points * [-1.0, -1.0, 1.0]


@app.cls(gpu="L40S", volumes={WEIGHTS: volume}, timeout=1800, scaledown_window=120)
class Sam3D:
    @modal.enter()
    def load(self):
        from inference import Inference

        self.inference = Inference(str(CKPT / "pipeline.yaml"), compile=False)

    @modal.method()
    def reconstruct(self, rgb_png: bytes, mask_png: bytes, pointmap_npy: bytes | None = None, seed: int = 42,
                    faces: int = 60000) -> dict:
        """One part. ``pointmap_npy``: HxWx3 float32, metres, OpenCV camera frame, NaN where unknown
        (any resolution; it is resized to the image). Returns the mesh in that camera frame and in
        SAM 3D's own object frame (same faces and vertex colours), and the layout between them."""
        import numpy as np
        import open3d as o3d
        import torch
        from PIL import Image
        from pytorch3d.transforms import quaternion_to_matrix

        rgb = np.array(Image.open(io.BytesIO(rgb_png)).convert("RGB"))
        mask = np.array(Image.open(io.BytesIO(mask_png)).convert("L")) > 127
        rgba = self.inference.merge_mask_to_rgba(rgb, mask)
        pointmap = None
        if pointmap_npy is not None:
            pm = torch.from_numpy(np.load(io.BytesIO(pointmap_npy)).astype(np.float32))
            pointmap = _p3d(pm)  # the pipeline works in PyTorch3D's camera convention
        out = self.inference._pipeline.run(
            rgba, None, seed, stage1_only=False, with_mesh_postprocess=False, with_texture_baking=False,
            with_layout_postprocess=pointmap is not None, use_vertex_color=True, pointmap=pointmap)

        glb = out["glb"]
        # to_glb turned the mesh from SAM 3D's z-up object frame to y-up; turn it back, then
        # place it with the layout (scale, rotation, translation: object -> camera).
        local = np.asarray(glb.vertices) @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]).T
        colors = np.asarray(glb.visual.vertex_colors)[:, :3] / 255.0
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(local), o3d.utility.Vector3iVector(np.asarray(glb.faces)))
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        if len(mesh.triangles) > faces:
            mesh = mesh.simplify_quadric_decimation(faces)
        mesh.remove_unreferenced_vertices()
        local = np.asarray(mesh.vertices)

        s = out["scale"].reshape(-1, 3)[0].float().cpu()
        r = quaternion_to_matrix(out["rotation"].reshape(-1, 4)[0].float().cpu())
        t = out["translation"].reshape(-1, 3)[0].float().cpu()
        # Transform3d().scale(s).rotate(R).translate(t), row vectors: p * s @ R + t
        cam = _p3d((torch.from_numpy(local).float() * s) @ r + t).numpy()
        return {"vertices_camera": cam.astype(np.float32), "vertices_local": local.astype(np.float32),
                "faces": np.asarray(mesh.triangles, np.int32),
                "colors": (np.clip(np.asarray(mesh.vertex_colors), 0, 1) * 255).astype(np.uint8),
                "scale": s.tolist(), "rotation": r.tolist(), "translation": t.tolist(),
                "layout_iou": float(out["iou"]) if "iou" in out else None}


@app.local_entrypoint()
def main():
    print(check_imports.remote())
