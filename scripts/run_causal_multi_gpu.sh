#!/bin/bash
set -euo pipefail

# ============================================================
# Multi-GPU Parallel Inference Script (Causal-Forcing)
# Core: inference.py with prompt sharding across independent processes
# Splits prompts across GPUs, each GPU runs inference independently
# ============================================================

OUTPUT_DIR=""
CONFIG_PATH="configs/causal_forcing_dmd_pyramid.yaml"
CHECKPOINT_PATH="checkpoints/chunkwise/causal_forcing.pt"
NUM_FRAMES=21
PROMPT_FILE="prompts/demos.txt"
GPUS=""
SEED=0
USE_EMA=""

usage() {
    cat <<EOF
Usage: bash $(basename "$0") -o OUTPUT_DIR [OPTIONS]

Required:
  -o, --output_dir DIR       Output video directory

Options:
  -c, --config PATH          YAML config path (default: $CONFIG_PATH)
  -p, --checkpoint PATH      Checkpoint path (default: $CHECKPOINT_PATH)
  -n, --num_frames N         Number of output latent frames (default: $NUM_FRAMES)
  -d, --prompts PATH         Prompt file (default: $PROMPT_FILE)
  -g, --gpus IDS             Comma-separated GPU IDs (default: auto-detect)
  -s, --seed N               Random seed (default: $SEED)
      --use_ema              Use EMA weights
  -h, --help                 Show this help
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -o|--output_dir)    OUTPUT_DIR="$2"; shift 2 ;;
        -c|--config)        CONFIG_PATH="$2"; shift 2 ;;
        -p|--checkpoint)    CHECKPOINT_PATH="$2"; shift 2 ;;
        -n|--num_frames)    NUM_FRAMES="$2"; shift 2 ;;
        -d|--prompts)       PROMPT_FILE="$2"; shift 2 ;;
        -g|--gpus)          GPUS="$2"; shift 2 ;;
        -s|--seed)          SEED="$2"; shift 2 ;;
        --use_ema)          USE_EMA="--use_ema"; shift ;;
        -h|--help)          usage ;;
        *)                  echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -z "$OUTPUT_DIR" ]; then
    echo "Error: --output_dir (-o) is required"
    echo "Run with --help for usage"
    exit 1
fi

if [ ! -f "$PROMPT_FILE" ]; then
    echo "Error: prompt file not found: $PROMPT_FILE"
    exit 1
fi

if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: config file not found: $CONFIG_PATH"
    exit 1
fi

if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Error: checkpoint not found: $CHECKPOINT_PATH"
    exit 1
fi

if [ -z "$GPUS" ]; then
    GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ',' | sed 's/,$//')
fi

IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
NUM_GPUS=${#GPU_ARRAY[@]}

if [ "$NUM_GPUS" -eq 0 ]; then
    echo "Error: no GPUs resolved"
    exit 1
fi

CLEAN_PROMPTS=$(mktemp /tmp/causal_prompts_XXXXXX.txt)
grep '.' "$PROMPT_FILE" > "$CLEAN_PROMPTS" || true
TOTAL=$(wc -l < "$CLEAN_PROMPTS")

if [ "$TOTAL" -eq 0 ]; then
    echo "Error: no prompts found in $PROMPT_FILE"
    rm -f "$CLEAN_PROMPTS"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

{
    echo "index,prompt"
    IDX=0
    while IFS= read -r line; do
        escaped="${line//\"/\"\"}"
        echo "${IDX},\"${escaped}\""
        IDX=$((IDX + 1))
    done < "$CLEAN_PROMPTS"
} > "$OUTPUT_DIR/prompts.csv"

echo "============================================================"
echo "Multi-GPU Causal-Forcing Inference"
echo "  Config:     $CONFIG_PATH"
echo "  Checkpoint: $CHECKPOINT_PATH"
echo "  Prompts:    $PROMPT_FILE ($TOTAL prompts)"
echo "  GPUs:       $GPUS ($NUM_GPUS GPUs)"
echo "  Frames:     $NUM_FRAMES latent frames"
echo "  Output:     $OUTPUT_DIR"
echo "============================================================"

PER_GPU=$((TOTAL / NUM_GPUS))

PIDS=()
TEMP_FILES=("$CLEAN_PROMPTS")

cleanup() {
    echo ""
    echo "Interrupted, killing all inference processes..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
    for f in "${TEMP_FILES[@]}"; do
        rm -f "$f"
    done
    exit 1
}
trap cleanup INT TERM

START_INDEX=0
START_LINE=1

for i in "${!GPU_ARRAY[@]}"; do
    GPU="${GPU_ARRAY[$i]}"
    if [ "$i" -eq $((NUM_GPUS - 1)) ]; then
        COUNT=$((TOTAL - PER_GPU * i))
    else
        COUNT=$PER_GPU
    fi

    if [ "$COUNT" -le 0 ]; then
        continue
    fi

    END_LINE=$((START_LINE + COUNT - 1))
    TEMP_FILE=$(mktemp /tmp/causal_gpu${GPU}_XXXXXX.txt)
    TEMP_FILES+=("$TEMP_FILE")
    sed -n "${START_LINE},${END_LINE}p" "$CLEAN_PROMPTS" > "$TEMP_FILE"

    echo "  GPU $GPU: prompts $START_INDEX-$((START_INDEX + COUNT - 1)) ($COUNT prompts)"

    CUDA_VISIBLE_DEVICES="$GPU" uv run python inference.py \
        --config_path "$CONFIG_PATH" \
        --checkpoint_path "$CHECKPOINT_PATH" \
        --data_path "$TEMP_FILE" \
        --output_folder "$OUTPUT_DIR" \
        --num_output_frames "$NUM_FRAMES" \
        --seed "$SEED" \
        --save_with_index \
        --start_index "$START_INDEX" \
        $USE_EMA &

    PIDS+=($!)

    START_INDEX=$((START_INDEX + COUNT))
    START_LINE=$((END_LINE + 1))
done

echo ""
echo "All GPU processes launched. Waiting for completion..."

FAIL=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || FAIL=1
done

for f in "${TEMP_FILES[@]}"; do
    rm -f "$f"
done

if [ "$FAIL" -ne 0 ]; then
    echo "Error: one or more GPU processes failed"
    exit 1
fi

VIDEO_COUNT=$(find "$OUTPUT_DIR" -maxdepth 1 -name 'video_*.mp4' | wc -l)
echo "============================================================"
echo "Done! Generated ${VIDEO_COUNT}/${TOTAL} videos in ${OUTPUT_DIR}"
echo "Prompt index: ${OUTPUT_DIR}/prompts.csv"
echo "============================================================"
