#!/usr/bin/env bash
#SBATCH --job-name=llmdrop
#SBATCH --partition=gpuqs               # change to your cluster's GPU partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00               # wall time: adjust per model size
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-user=ashley.irawan@sjsu.edu
#SBATCH --mail-type=END

export http_proxy=http://172.16.1.2:3128
export https_proxy=http://172.16.1.2:3128

module load python3/3.12.12
module load ml/torch/2.6

source ~/venvs/llmdrop/bin/activate

# ─── CONFIG ────────────────────────────────────────────────────────────────────

PORT="21304"
GPUs="0"

# Model
MODEL_NAME="llama3-8b"
MODEL_NAME_OR_PATH="meta-llama/Meta-Llama-3-8B"   # HF hub id or local path

# Calibration
DATASET="c4_val"
PRUNE_DATA_TYPE="pt"
N_CALIBRATION_SAMPLES=256
SEQ_LEN=2048

# Pruning method: "block_drop" or "layer_drop"
PRUNE_METHOD="layer_drop"

# --- block_drop settings (used when PRUNE_METHOD=block_drop) ---
BLOCK_DROP_METHOD="discrete"   # "discrete" or "consecutive"
DROP_N=12

# --- layer_drop settings (used when PRUNE_METHOD=layer_drop) ---
LAYER_DROP_METHOD="discrete"
TARGET_LAYER="attn"            # "attn", "mlp", or "all"
LAYER_DROP_NORM=True
ONLY_UPDATE_CONFIG=False       # True = skip re-saving weights (saves disk)

# Evaluation tasks and their few-shot counts (parallel arrays)
TASKS=("boolq" "rte" "openbookqa" "piqa" "mmlu" "winogrande" "gsm8k" "hellaswag" "arc_challenge")
NUM_FEWSHOTS=("0"   "0"   "0"          "0"    "5"    "5"          "5"     "10"        "25")

# ─── ENVIRONMENT ───────────────────────────────────────────────────────────────

# Adjust or remove these module/conda lines to match your cluster setup.
# module load cuda/12.1
# module load anaconda/2023.09
# conda activate llmdrop

set -euo pipefail

# ─── DERIVED PATHS ─────────────────────────────────────────────────────────────

cd ~/LLM-Drop-v2
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ "$PRUNE_METHOD" == "block_drop" ]]; then
    FOLDER_NAME="${MODEL_NAME}-${PRUNE_METHOD}-${BLOCK_DROP_METHOD}-drop${DROP_N}"
    CACHE_SUFFIX="${PRUNE_METHOD}-${DATASET}-${N_CALIBRATION_SAMPLES}samples"
elif [[ "$PRUNE_METHOD" == "layer_drop" ]]; then
    FOLDER_NAME="${MODEL_NAME}-${PRUNE_METHOD}_${TARGET_LAYER}-${LAYER_DROP_METHOD}-drop${DROP_N}"
    CACHE_SUFFIX="${PRUNE_METHOD}_${TARGET_LAYER}-${DATASET}-${N_CALIBRATION_SAMPLES}samples"
fi

OUTPUT_DIR="../results_prune/${FOLDER_NAME}"
PRUNE_SAVE_PATH="${OUTPUT_DIR}/checkpoint"
SIMILARITY_CACHE_FILE="../results_prune/cache/${MODEL_NAME}-${CACHE_SUFFIX}.pt"
EVAL_OUTPUT_DIR="${OUTPUT_DIR}/eval"

mkdir -p "${OUTPUT_DIR}" "../results_prune/cache" "${EVAL_OUTPUT_DIR}" logs

echo "========================================"
echo "  Job        : ${SLURM_JOB_ID}"
echo "  Node       : ${SLURM_NODELIST}"
echo "  Method     : ${PRUNE_METHOD}"
echo "  Model      : ${MODEL_NAME_OR_PATH}"
echo "  Output     : ${PRUNE_SAVE_PATH}"
echo "========================================"

# ─── STEP 1: CALIBRATION + DROP DECISION ───────────────────────────────────────

echo ""
echo "[1/3] Calibration + drop decision..."

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
        --prune_model_save_path "${PRUNE_SAVE_PATH}"

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
        --prune_model_save_path "${PRUNE_SAVE_PATH}"
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
        --prune_model_save_path "${PRUNE_SAVE_PATH}"

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
        --prune_model_save_path "${PRUNE_SAVE_PATH}"
fi

# ─── STEP 3: lm_eval BENCHMARKS ────────────────────────────────────────────────

echo ""
echo "[3/3] Running lm_eval benchmarks..."

NUM_TASKS=${#TASKS[@]}
for ((i=0; i<NUM_TASKS; i++)); do
    TASK="${TASKS[$i]}"
    FEWSHOT="${NUM_FEWSHOTS[$i]}"
    RESULT_FILE="${EVAL_OUTPUT_DIR}/${FEWSHOT}shot_${TASK}.json"

    echo "  -> ${TASK} (${FEWSHOT}-shot)"

    accelerate launch --num_processes 1 --main_process_port $PORT \
        -m lm_eval \
        --model hf \
        --model_args "pretrained=${PRUNE_SAVE_PATH},trust_remote_code=True,dtype=float16" \
        --tasks "${TASK}" \
        --num_fewshot "${FEWSHOT}" \
        --batch_size 1 \
        --output_path "${RESULT_FILE}" \
        >> "${EVAL_OUTPUT_DIR}/eval_${TASK}.out" 2>&1

    echo "     saved -> ${RESULT_FILE}"
done

echo ""
echo "========================================"
echo "  Done."
echo "  Pruned model : ${PRUNE_SAVE_PATH}"
echo "  Eval results : ${EVAL_OUTPUT_DIR}/"
echo "========================================"
