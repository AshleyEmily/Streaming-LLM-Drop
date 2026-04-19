#!/usr/bin/env bash
#SBATCH --job-name=llmdrop-bench-mistral
#SBATCH --partition=gpuqs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=2:00:00
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

PROMPT_LEN=2048
GEN_LEN=2048
RUNS=3

STREAMING_GEN_LEN=2048

BASELINE_PATH="mistralai/Mistral-7B-v0.1"
MODEL_NAME="mistral-base"

# ─── SETUP ─────────────────────────────────────────────────────────────────────

cd ~/LLM-Drop-v2
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
RESULTS_DIR="$(realpath ../results_prune)"

# Standard (non-streaming) checkpoints
STANDARD_CHECKPOINTS=(
    "${RESULTS_DIR}/${MODEL_NAME}-layer_drop_attn-discrete-drop8/checkpoint"
)

# Streaming checkpoints as parallel arrays: path, n_init, n_local
STREAMING_CHECKPOINTS=(
    "${RESULTS_DIR}/${MODEL_NAME}-streamllm-nodrop/checkpoint"
    "${RESULTS_DIR}/${MODEL_NAME}-layer_drop_attn-discrete-drop8-streamllm-4092-4/checkpoint"
)
STREAMING_N_INIT=(4    4)
STREAMING_N_LOCAL=(4092 4092)

mkdir -p logs

echo "========================================"
echo "  Job        : ${SLURM_JOB_ID}"
echo "  Node       : ${SLURM_NODELIST}"
echo "  Prompt len : ${PROMPT_LEN}"
echo "  Gen len    : ${GEN_LEN}"
echo "  Runs       : ${RUNS}"
echo "========================================"

# ─── STANDARD BENCHMARKS ───────────────────────────────────────────────────────

echo ""
echo "[Baseline — standard] ${BASELINE_PATH}"
python scripts/benchmark_inference.py \
    --model_path "${BASELINE_PATH}" \
    --prompt_len "${PROMPT_LEN}" \
    --gen_len    "${GEN_LEN}" \
    --runs       "${RUNS}"

for CKPT in "${STANDARD_CHECKPOINTS[@]}"; do
    echo ""
    echo "[Standard] ${CKPT}"
    python scripts/benchmark_inference.py \
        --model_path "${CKPT}" \
        --prompt_len "${PROMPT_LEN}" \
        --gen_len    "${GEN_LEN}" \
        --runs       "${RUNS}"
done

# ─── STREAMING BENCHMARKS ──────────────────────────────────────────────────────

NUM_STREAMING=${#STREAMING_CHECKPOINTS[@]}
for ((i=0; i<NUM_STREAMING; i++)); do
    CKPT="${STREAMING_CHECKPOINTS[$i]}"
    N_INIT="${STREAMING_N_INIT[$i]}"
    N_LOCAL="${STREAMING_N_LOCAL[$i]}"
    echo ""
    echo "[Streaming] ${CKPT} (n_init=${N_INIT}, n_local=${N_LOCAL})"
    python scripts/benchmark_inference.py \
        --model_path  "${CKPT}" \
        --streaming \
        --start_size  "${N_INIT}" \
        --recent_size "${N_LOCAL}" \
        --prompt_len  "${PROMPT_LEN}" \
        --gen_len     "${STREAMING_GEN_LEN}" \
        --runs        "${RUNS}"
done

echo ""
echo "========================================"
echo "  Done."
echo "========================================"
