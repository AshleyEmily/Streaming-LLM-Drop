#!/usr/bin/env bash
#SBATCH --job-name=llmdrop-bench
#SBATCH --partition=gpuqs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:h100:1
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

PROMPT_LEN=512
GEN_LEN=128
RUNS=3

# Streaming benchmark uses a longer gen_len to demonstrate bounded KV memory —
# the key thesis result: memory stays flat at (start_size + recent_size) regardless of gen_len.
STREAMING_GEN_LEN=4096
STREAMLLM_N_INIT=128
STREAMLLM_N_LOCAL=4096

BASELINE_PATH="mistralai/Mistral-7B-v0.1"

# Standard (non-streaming) checkpoints
STANDARD_CHECKPOINTS=(
    "../results_prune/mistral-base-block_drop-discrete-drop8/checkpoint"
    "../results_prune/mistral-base-layer_drop_attn-discrete-drop8/checkpoint"
    "../results_prune/mistral-base-ntk_rope-nodrop/checkpoint"
    "../results_prune/mistral-base-gqa-nodrop/checkpoint"
)

# Streaming checkpoints — run with enable_streaming_llm (--streaming flag).
# The streamllm-nodrop baseline uses the base model directly: loading the nodrop
# checkpoint would layer the calibration-style mask patch under enable_streaming_llm.
STREAMING_CHECKPOINTS=(
    # nodrop baseline: base model + streaming inference
    "${BASELINE_PATH}"
    # pruned with streamllm calibration + streaming inference
    "../results_prune/mistral-base-layer_drop_attn-discrete-drop8-streamllm/checkpoint"
)

# ─── SETUP ─────────────────────────────────────────────────────────────────────

cd ~/LLM-Drop-v2
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"

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
# Uses longer gen_len to show that KV memory stays bounded at
# (start_size + recent_size) tokens regardless of generation length.

for CKPT in "${STREAMING_CHECKPOINTS[@]}"; do
    echo ""
    echo "[Streaming] ${CKPT}"
    python scripts/benchmark_inference.py \
        --model_path  "${CKPT}" \
        --streaming \
        --start_size  "${STREAMLLM_N_INIT}" \
        --recent_size "${STREAMLLM_N_LOCAL}" \
        --prompt_len  "${PROMPT_LEN}" \
        --gen_len     "${STREAMING_GEN_LEN}" \
        --runs        "${RUNS}"
done

echo ""
echo "========================================"
echo "  Done."
echo "========================================"
