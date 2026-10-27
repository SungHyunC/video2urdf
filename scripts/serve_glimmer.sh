#!/usr/bin/env bash
# Serve Muse Glimmer (open weights, Apache-2.0) behind an OpenAI-compatible API on the A100 server.
#
#   pip install "vllm>=0.28"          # in its own venv on the server
#   bash scripts/serve_glimmer.sh     # binds 127.0.0.1 only; reach it through the SSH tunnel
#
# Desktop side: `ssh -L 8000:localhost:8000 <host>`, set V2U_PROFILE=local in .env, then `v2u llm-check`.
# Overrides: MODEL=<hf repo>, MAX_LEN=<tokens>, PORT=<port>.
set -euo pipefail

PORT="${PORT:-8000}"
VRAM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)

if [[ -z "${MODEL:-}" ]]; then
  if (( VRAM_MIB >= 75000 )); then
    # bf16 weights are ~60 GB; Meta budgets 72 GB of VRAM to serve.
    MODEL=meta-models/Muse-Glimmer-30B
    MAX_LEN="${MAX_LEN:-32768}"
  else
    # A100 40GB: third-party FP8 build (32.8 GB). Ampere has no FP8 compute, so vLLM runs it weight-only
    # (W8A16 Marlin). Untested on A100; leaves ~6 GB for KV cache, hence the short context.
    MODEL=RedHatAI/Muse-Glimmer-30B-FP8-block
    MAX_LEN="${MAX_LEN:-16384}"
  fi
fi
MAX_LEN="${MAX_LEN:-32768}"

echo "Serving ${MODEL} as 'muse-glimmer' on 127.0.0.1:${PORT} (max-model-len ${MAX_LEN}, GPU ${VRAM_MIB} MiB)"
exec vllm serve "${MODEL}" \
  --served-model-name muse-glimmer \
  --host 127.0.0.1 --port "${PORT}" \
  --max-model-len "${MAX_LEN}" \
  --enable-auto-tool-choice --tool-call-parser muse_glimmer \
  --reasoning-parser muse_glimmer \
  --generation-config auto
