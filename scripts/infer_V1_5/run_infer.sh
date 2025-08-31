#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/home/jusheng/yijia/LLaVA_HCP/checkpoints/llava-v1.5-7b-sft-stage2/checkpoint-45000}"
IMAGE_PATH="${IMAGE_PATH:-/home/jusheng/yijia/LLaVA_HCP/images/llava_logo.png}"
PROMPT="${PROMPT:-decirblr this image}"
DEVICE="${DEVICE:-cuda}"
TEMP="${TEMP:-0.2}"
TOP_P="${TOP_P:-}"
NUM_BEAMS="${NUM_BEAMS:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"

SCRIPT_DIR="/home/jusheng/yijia/LLaVA_HCP/scripts/infer_V1_5"

CMD=(
  python "$SCRIPT_DIR/infer_v1_5.py"
  --model-path "$MODEL_PATH"
  --image "$IMAGE_PATH"
  --prompt "$PROMPT"
  --device "$DEVICE"
  --temperature "$TEMP"
  --num_beams "$NUM_BEAMS"
  --max_new_tokens "$MAX_NEW_TOKENS"
)

if [[ -n "${TOP_P}" ]]; then
  CMD+=(--top_p "$TOP_P")
fi

"${CMD[@]}"


