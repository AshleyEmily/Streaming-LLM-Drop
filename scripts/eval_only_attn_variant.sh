#!/usr/bin/env bash
#SBATCH --job-name=llmdrop-eval-attn
#SBATCH --partition=gpuqs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-user=ashley.irawan@sjsu.edu
#SBATCH --mail-type=END

export http_proxy=http://172.16.1.2:3128
export https_proxy=http://172.16.1.2:3128

module load python3/3.12.12
module load ml/torch/2.6

source ~/venvs/llmdrop/bin/activate

set -euo pipefail

# ─── CONFIG ────────────────────────────────────────────────────────────────────

PORT="21304"

MODEL_NAME="mistral-base"
PRUNE_METHOD="layer_drop"

# Attention variant: "streamllm" | "ntk_rope" | "gqa"
ATTENTION_VARIANT="streamllm"

# block_drop settings (used when PRUNE_METHOD=block_drop)
BLOCK_DROP_METHOD="discrete"
DROP_N=8

# layer_drop settings (used when PRUNE_METHOD=layer_drop)
LAYER_DROP_METHOD="discrete"
TARGET_LAYER="attn"

TASKS=("boolq" "rte" "openbookqa" "piqa" "mmlu" "winogrande" "gsm8k" "hellaswag" "arc_challenge")
NUM_FEWSHOTS=("0"   "0"   "0"          "0"    "5"    "5"          "5"     "10"        "25")

# ─── DERIVED PATHS ─────────────────────────────────────────────────────────────

cd ~/LLM-Drop-v2
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ "$PRUNE_METHOD" == "block_drop" ]]; then
    FOLDER_NAME="${MODEL_NAME}-${PRUNE_METHOD}-${BLOCK_DROP_METHOD}-drop${DROP_N}-${ATTENTION_VARIANT}"
elif [[ "$PRUNE_METHOD" == "layer_drop" ]]; then
    FOLDER_NAME="${MODEL_NAME}-${PRUNE_METHOD}_${TARGET_LAYER}-${LAYER_DROP_METHOD}-drop${DROP_N}-${ATTENTION_VARIANT}"
fi

OUTPUT_DIR="../results_prune/${FOLDER_NAME}"
PRUNE_SAVE_PATH="${OUTPUT_DIR}/checkpoint"
EVAL_OUTPUT_DIR="${OUTPUT_DIR}/eval"

mkdir -p "${EVAL_OUTPUT_DIR}" logs

echo "========================================"
echo "  Job        : ${SLURM_JOB_ID}"
echo "  Node       : ${SLURM_NODELIST}"
echo "  Variant    : ${ATTENTION_VARIANT}"
echo "  Model      : ${PRUNE_SAVE_PATH}"
echo "  Eval output: ${EVAL_OUTPUT_DIR}"
echo "========================================"

# ─── lm_eval BENCHMARKS ────────────────────────────────────────────────────────

NUM_TASKS=${#TASKS[@]}
for ((i=0; i<NUM_TASKS; i++)); do
    TASK="${TASKS[$i]}"
    FEWSHOT="${NUM_FEWSHOTS[$i]}"
    RESULT_FILE="${EVAL_OUTPUT_DIR}/${FEWSHOT}shot_${TASK}.json"
    PERF_LOG="${EVAL_OUTPUT_DIR}/perf_${TASK}.log"

    echo "  -> ${TASK} (${FEWSHOT}-shot)"

    # Poll GPU memory every second into perf log (background)
    nvidia-smi --query-gpu=memory.used,memory.free,memory.total \
        --format=csv,noheader,nounits --loop=1 \
        >> "${PERF_LOG}" 2>&1 &
    NVSMI_PID=$!

    TIME_START=$(date +%s)

    accelerate launch --num_processes 1 --main_process_port $PORT \
        -m lm_eval \
        --model hf \
        --model_args "pretrained=${PRUNE_SAVE_PATH},trust_remote_code=True,dtype=float16" \
        --tasks "${TASK}" \
        --num_fewshot "${FEWSHOT}" \
        --batch_size 1 \
        --output_path "${RESULT_FILE}" \
        >> "${EVAL_OUTPUT_DIR}/eval_${TASK}.out" 2>&1

    TIME_END=$(date +%s)
    kill "${NVSMI_PID}" 2>/dev/null || true

    ELAPSED=$((TIME_END - TIME_START))
    PEAK_MEM=$(awk -F',' '{gsub(/ /, "", $1); print $1+0}' "${PERF_LOG}" | sort -n | tail -1)
    echo "     saved  -> ${RESULT_FILE}"
    echo "     time   -> ${ELAPSED}s"
    echo "     peak   -> ${PEAK_MEM} MiB GPU memory"
done

echo ""
echo "========================================"
echo "  Done."
echo "  Eval results: ${EVAL_OUTPUT_DIR}/"
echo "========================================"
