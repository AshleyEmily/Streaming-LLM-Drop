#!/usr/bin/env bash
#SBATCH --job-name=llmdrop-attn-stream-2
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

# Model
MODEL_NAME="llama3-8b"
MODEL_NAME_OR_PATH="meta-llama/Meta-Llama-3-8B"

# Calibration
DATASET="c4_val"
PRUNE_DATA_TYPE="pt"
N_CALIBRATION_SAMPLES=64  # 128×8192 ≈ 256×4096 tokens; keeps arrays within 40 GB GPU

# Attention variant: "streamllm" | "ntk_rope" | "gqa"
ATTENTION_VARIANT="streamllm"

# StreamLLM settings (used when ATTENTION_VARIANT=streamllm)
# SEQ_LEN must exceed n_init + n_local for StreamLLM to differ from full attention
STREAMLLM_N_INIT=4
STREAMLLM_N_LOCAL=8188   # match Llama 3's full context window
SEQ_LEN=16384             # must be > 4+8192=8196; 2× context is the natural choice
# must be > STREAMLLM_N_INIT + STREAMLLM_N_LOCAL (4+8192=8196)

# NTK-RoPE settings (used when ATTENTION_VARIANT=ntk_rope)
NTK_ROPE_FACTOR=4.0

# GQA settings (used when ATTENTION_VARIANT=gqa)
GQA_NUM_KV_HEADS=4

# Pruning method: "block_drop" or "layer_drop"
PRUNE_METHOD="layer_drop"

# --- block_drop settings (used when PRUNE_METHOD=block_drop) ---
BLOCK_DROP_METHOD="discrete"   # "discrete" or "consecutive"
DROP_N=12

# --- layer_drop settings (used when PRUNE_METHOD=layer_drop) ---
LAYER_DROP_METHOD="discrete"
TARGET_LAYER="attn"            # "attn", "mlp", or "all"
LAYER_DROP_NORM=True
ONLY_UPDATE_CONFIG=False

# Evaluation tasks and their few-shot counts (parallel arrays)
TASKS=("boolq" "rte" "openbookqa" "piqa" "mmlu" "winogrande" "gsm8k" "hellaswag" "arc_challenge")
NUM_FEWSHOTS=("0"   "0"   "0"          "0"    "5"    "5"          "5"     "10"        "25")

# ─── DERIVED PATHS ─────────────────────────────────────────────────────────────

cd ~/LLM-Drop-v2
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ "$PRUNE_METHOD" == "block_drop" ]]; then
    FOLDER_NAME="${MODEL_NAME}-${PRUNE_METHOD}-${BLOCK_DROP_METHOD}-drop${DROP_N}-${ATTENTION_VARIANT}-${STREAMLLM_N_LOCAL}-${STREAMLLM_N_INIT}"
    CACHE_SUFFIX="${PRUNE_METHOD}-${DATASET}-${N_CALIBRATION_SAMPLES}samples-${ATTENTION_VARIANT}-${STREAMLLM_N_LOCAL}-${STREAMLLM_N_INIT}"
elif [[ "$PRUNE_METHOD" == "layer_drop" ]]; then
    FOLDER_NAME="${MODEL_NAME}-${PRUNE_METHOD}_${TARGET_LAYER}-${LAYER_DROP_METHOD}-drop${DROP_N}-${ATTENTION_VARIANT}-${STREAMLLM_N_LOCAL}-${STREAMLLM_N_INIT}"
    CACHE_SUFFIX="${PRUNE_METHOD}_${TARGET_LAYER}-${DATASET}-${N_CALIBRATION_SAMPLES}samples-${ATTENTION_VARIANT}-${STREAMLLM_N_LOCAL}-${STREAMLLM_N_INIT}"
fi

OUTPUT_DIR="../results_prune/${FOLDER_NAME}"
PRUNE_SAVE_PATH="${OUTPUT_DIR}/checkpoint"
SIMILARITY_CACHE_FILE="../results_prune/cache/${MODEL_NAME}-${CACHE_SUFFIX}-${STREAMLLM_N_LOCAL}-${STREAMLLM_N_INIT}.pt"
EVAL_OUTPUT_DIR="${OUTPUT_DIR}/eval"

mkdir -p "${OUTPUT_DIR}" "../results_prune/cache" "${EVAL_OUTPUT_DIR}" logs

echo "========================================"
echo "  Job        : ${SLURM_JOB_ID}"
echo "  Node       : ${SLURM_NODELIST}"
echo "  Method     : ${PRUNE_METHOD}"
echo "  Attn variant: ${ATTENTION_VARIANT}"
echo "  Model      : ${MODEL_NAME_OR_PATH}"
echo "  DROP_N          : ${DROP_N}"
echo "  STREAMLLM_N_INIT : ${STREAMLLM_N_INIT}"
echo "  STREAMLLM_N_LOCAL: ${STREAMLLM_N_LOCAL}"
echo "  SEQ_LEN          : ${SEQ_LEN}"
echo "  Output     : ${PRUNE_SAVE_PATH}"
echo "========================================"

# ─── BUILD ATTENTION VARIANT ARGS ──────────────────────────────────────────────

ATTN_ARGS="--attention_variant ${ATTENTION_VARIANT}"

if [[ "$ATTENTION_VARIANT" == "streamllm" ]]; then
    ATTN_ARGS="${ATTN_ARGS} --streamllm_n_init ${STREAMLLM_N_INIT} --streamllm_n_local ${STREAMLLM_N_LOCAL}"
elif [[ "$ATTENTION_VARIANT" == "ntk_rope" ]]; then
    ATTN_ARGS="${ATTN_ARGS} --ntk_rope_factor ${NTK_ROPE_FACTOR}"
elif [[ "$ATTENTION_VARIANT" == "gqa" ]]; then
    ATTN_ARGS="${ATTN_ARGS} --gqa_num_kv_heads ${GQA_NUM_KV_HEADS}"
fi

# ─── STEP 1: CALIBRATION + DROP DECISION ───────────────────────────────────────

echo ""
echo "[1/3] Calibration + drop decision (attention_variant=${ATTENTION_VARIANT})..."

if [[ "$PRUNE_METHOD" == "block_drop" ]]; then
    accelerate launch --num_processes 1 --main_process_port $PORT \
        src/compress.py \
        --stage prune \
        --model_name_or_path "${MODEL_NAME_OR_PATH}" \
        --dataset "${DATASET}" \
        --dataset_dir ./src/llmtuner/data \
        --split "train" \
        --prune_data_type "${PRUNE_DATA_TYPE}" \
        --cutoff_len "${SEQ_LEN}" \
        --output_dir "${OUTPUT_DIR}" \
        --logging_steps 10 \
        --fp16 \
        --n_calibration_samples "${N_CALIBRATION_SAMPLES}" \
        --prune_method "${PRUNE_METHOD}" \
        --block_drop_method "${BLOCK_DROP_METHOD}" \
        --drop_n "${DROP_N}" \
        --similarity_cache_file "${SIMILARITY_CACHE_FILE}" \
        --prune_model_save_path "${PRUNE_SAVE_PATH}" \
        ${ATTN_ARGS}

elif [[ "$PRUNE_METHOD" == "layer_drop" ]]; then
    accelerate launch --num_processes 1 --main_process_port $PORT \
        src/compress.py \
        --stage prune \
        --model_name_or_path "${MODEL_NAME_OR_PATH}" \
        --dataset "${DATASET}" \
        --dataset_dir ./src/llmtuner/data \
        --split "train" \
        --prune_data_type "${PRUNE_DATA_TYPE}" \
        --cutoff_len "${SEQ_LEN}" \
        --layer_drop_norm "${LAYER_DROP_NORM}" \
        --target_layer "${TARGET_LAYER}" \
        --only_update_config "${ONLY_UPDATE_CONFIG}" \
        --output_dir "${OUTPUT_DIR}" \
        --logging_steps 10 \
        --fp16 \
        --n_calibration_samples "${N_CALIBRATION_SAMPLES}" \
        --prune_method "${PRUNE_METHOD}" \
        --layer_drop_method "${LAYER_DROP_METHOD}" \
        --drop_n "${DROP_N}" \
        --similarity_cache_file "${SIMILARITY_CACHE_FILE}" \
        --prune_model_save_path "${PRUNE_SAVE_PATH}" \
        ${ATTN_ARGS}
fi

# ─── STEP 2: POST-DROPPING ─────────────────────────────────────────────────────

echo ""
echo "[2/3] Post-dropping: saving pruned model..."

if [[ "$PRUNE_METHOD" == "block_drop" ]]; then
    python src/compress.py \
        --stage prune \
        --model_name_or_path "${MODEL_NAME_OR_PATH}" \
        --dataset "${DATASET}" \
        --dataset_dir ./src/llmtuner/data \
        --split "train" \
        --only_update_config "${ONLY_UPDATE_CONFIG}" \
        --prune_data_type "${PRUNE_DATA_TYPE}" \
        --cutoff_len "${SEQ_LEN}" \
        --output_dir "${OUTPUT_DIR}" \
        --logging_steps 10 \
        --fp16 \
        --n_calibration_samples "${N_CALIBRATION_SAMPLES}" \
        --prune_method "${PRUNE_METHOD}" \
        --block_drop_method "post_dropping" \
        --drop_n "${DROP_N}" \
        --similarity_cache_file "${SIMILARITY_CACHE_FILE}" \
        --prune_model_save_path "${PRUNE_SAVE_PATH}" \
        ${ATTN_ARGS}

elif [[ "$PRUNE_METHOD" == "layer_drop" ]]; then
    python src/compress.py \
        --stage prune \
        --model_name_or_path "${MODEL_NAME_OR_PATH}" \
        --dataset "${DATASET}" \
        --dataset_dir ./src/llmtuner/data \
        --split "train" \
        --only_update_config "${ONLY_UPDATE_CONFIG}" \
        --layer_drop_norm "${LAYER_DROP_NORM}" \
        --target_layer "${TARGET_LAYER}" \
        --prune_data_type "${PRUNE_DATA_TYPE}" \
        --cutoff_len "${SEQ_LEN}" \
        --output_dir "${OUTPUT_DIR}" \
        --logging_steps 10 \
        --fp16 \
        --n_calibration_samples "${N_CALIBRATION_SAMPLES}" \
        --prune_method "${PRUNE_METHOD}" \
        --layer_drop_method "post_dropping" \
        --drop_n "${DROP_N}" \
        --similarity_cache_file "${SIMILARITY_CACHE_FILE}" \
        --prune_model_save_path "${PRUNE_SAVE_PATH}" \
        ${ATTN_ARGS}
fi

# ─── STEP 3: lm_eval BENCHMARKS ────────────────────────────────────────────────

echo ""
echo "[3/3] Running lm_eval benchmarks..."

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
echo "  Pruned model : ${PRUNE_SAVE_PATH}"
echo "  Eval results : ${EVAL_OUTPUT_DIR}/"
echo "========================================"
