#!/usr/bin/env bash
#SBATCH --job-name=llmdrop-ppl-2
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

# Full (unpruned) base model for comparison
FULL_MODEL_PATH="mistralai/Mistral-7B-v0.1"
MODEL_NAME="mistral-base"

# Pruned streamllm model settings (must match prune_attention_variant.sh)
PRUNE_METHOD="layer_drop"
BLOCK_DROP_METHOD="discrete"
DROP_N=8
LAYER_DROP_METHOD="discrete"
TARGET_LAYER="attn"
ATTENTION_VARIANT="streamllm"

# StreamingLLM eval settings (matching original streaming-llm paper config)
START_SIZE=4
RECENT_SIZE=4092
NUM_EVAL_TOKENS=400000

WINDOW_SIZE=${RECENT_SIZE}   # must match STREAMLLM_N_LOCAL in prune_attention_variant.sh

# ─── DERIVED PATHS ─────────────────────────────────────────────────────────────

cd ~/LLM-Drop-v2
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ "$PRUNE_METHOD" == "block_drop" ]]; then
    FOLDER_NAME="${MODEL_NAME}-${PRUNE_METHOD}-${BLOCK_DROP_METHOD}-drop${DROP_N}-${ATTENTION_VARIANT}-${WINDOW_SIZE}-${START_SIZE}"
elif [[ "$PRUNE_METHOD" == "layer_drop" ]]; then
    FOLDER_NAME="${MODEL_NAME}-${PRUNE_METHOD}_${TARGET_LAYER}-${LAYER_DROP_METHOD}-drop${DROP_N}-${ATTENTION_VARIANT}-${WINDOW_SIZE}-${START_SIZE}"
fi
OUTPUT_DIR="../results_prune/${FOLDER_NAME}"
PRUNED_MODEL_PATH="${OUTPUT_DIR}/checkpoint"

FULL_PPL_DIR="${OUTPUT_DIR}/ppl_full_${SLURM_JOB_ID}"
PRUNED_PPL_DIR="${OUTPUT_DIR}/ppl_streaming_${SLURM_JOB_ID}"

mkdir -p "${FULL_PPL_DIR}" "${PRUNED_PPL_DIR}" logs

echo "========================================"
echo "  Job         : ${SLURM_JOB_ID}"
echo "  Node        : ${SLURM_NODELIST}"
echo "  Full model  : ${FULL_MODEL_PATH}"
echo "  Pruned model: ${PRUNED_MODEL_PATH}"
echo "  DROP_N      : ${DROP_N}"
echo "  start_size  : ${START_SIZE}"
echo "  recent_size : ${RECENT_SIZE}"
echo "  eval_tokens : ${NUM_EVAL_TOKENS}"
echo "  WINDOW_SIZE : ${WINDOW_SIZE}"
echo "========================================"

# ─── STEP 1: FULL MODEL ────────────────────────────────────────────────────────

echo ""
echo "[1/2] Evaluating full (unpruned) model..."

python scripts/eval_long_ppl.py \
    --model_path "${FULL_MODEL_PATH}" \
    --start_size "${START_SIZE}" \
    --recent_size "${RECENT_SIZE}" \
    --num_eval_tokens "${NUM_EVAL_TOKENS}" \
    --output_dir "${FULL_PPL_DIR}"

# ─── STEP 2: PRUNED STREAMLLM MODEL ───────────────────────────────────────────

echo ""
echo "[2/2] Evaluating pruned streamllm model..."

python scripts/eval_long_ppl.py \
    --model_path "${PRUNED_MODEL_PATH}" \
    --start_size "${START_SIZE}" \
    --recent_size "${RECENT_SIZE}" \
    --num_eval_tokens "${NUM_EVAL_TOKENS}" \
    --output_dir "${PRUNED_PPL_DIR}"

# ─── COMPARISON ───────────────────────────────────────────────────────────────

echo ""
echo "========================================"
echo "  COMPARISON"
echo "========================================"

parse_metric() {
    local file=$1 key=$2
    grep "^${key}=" "${file}" | cut -d= -f2
}

FULL_METRICS="${FULL_PPL_DIR}/metrics.txt"
PRUNED_METRICS="${PRUNED_PPL_DIR}/metrics.txt"

printf "%-28s %-18s %-18s\n" "Metric" "Full model" "Pruned+StreamLLM"
printf "%-28s %-18s %-18s\n" "------" "----------" "----------------"
printf "%-28s %-18s %-18s\n" "Perplexity"        "$(parse_metric $FULL_METRICS ppl)"               "$(parse_metric $PRUNED_METRICS ppl)"
printf "%-28s %-18s %-18s\n" "Throughput (tok/s)" "$(parse_metric $FULL_METRICS throughput_tok_per_s)" "$(parse_metric $PRUNED_METRICS throughput_tok_per_s)"
printf "%-28s %-18s %-18s\n" "Elapsed (s)"        "$(parse_metric $FULL_METRICS elapsed_s)"           "$(parse_metric $PRUNED_METRICS elapsed_s)"
printf "%-28s %-18s %-18s\n" "Model weights (MiB)" "$(parse_metric $FULL_METRICS model_mem_mib)"      "$(parse_metric $PRUNED_METRICS model_mem_mib)"
printf "%-28s %-18s %-18s\n" "Peak GPU mem (MiB)"  "$(parse_metric $FULL_METRICS peak_mem_mib)"       "$(parse_metric $PRUNED_METRICS peak_mem_mib)"
printf "%-28s %-18s %-18s\n" "KV cache mem (MiB)"  "$(parse_metric $FULL_METRICS kv_cache_mem_mib)"   "$(parse_metric $PRUNED_METRICS kv_cache_mem_mib)"

echo ""
echo "  Full results   : ${FULL_PPL_DIR}/metrics.txt"
echo "  Pruned results : ${PRUNED_PPL_DIR}/metrics.txt"
echo "========================================"
