#!/usr/bin/env bash
#SBATCH --job-name=llmdrop-variant-baseline
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

MODEL_NAME_OR_PATH="mistralai/Mistral-7B-v0.1"

STREAMLLM_N_INIT=128
STREAMLLM_N_LOCAL=4096
NTK_ROPE_FACTOR=4.0
GQA_NUM_KV_HEADS=4

# Inference benchmark settings (streaming nodrop baseline)
PROMPT_LEN=512
GEN_LEN=4096
BENCHMARK_RUNS=3

# Inference benchmark settings (streaming nodrop baseline)
PROMPT_LEN=512
GEN_LEN=4096
BENCHMARK_RUNS=3

# ─── SETUP ─────────────────────────────────────────────────────────────────────

cd ~/LLM-Drop-v2
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p logs

echo "========================================"
echo "  Job  : ${SLURM_JOB_ID}"
echo "  Node : ${SLURM_NODELIST}"
echo "  Model: ${MODEL_NAME_OR_PATH}"
echo "========================================"

# ─── StreamLLM baseline ────────────────────────────────────────────────────────

echo ""
echo "[1/3] StreamLLM baseline (n_init=${STREAMLLM_N_INIT}, n_local=${STREAMLLM_N_LOCAL})..."
python scripts/create_variant_baseline.py \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --attention_variant  streamllm \
    --streamllm_n_init   "${STREAMLLM_N_INIT}" \
    --streamllm_n_local  "${STREAMLLM_N_LOCAL}" \
    --output_dir ../results_prune/mistral-base-streamllm-nodrop/checkpoint

# ─── NTK-RoPE baseline ─────────────────────────────────────────────────────────

echo ""
echo "[2/3] NTK-RoPE baseline (factor=${NTK_ROPE_FACTOR})..."
python scripts/create_variant_baseline.py \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --attention_variant  ntk_rope \
    --ntk_rope_factor    "${NTK_ROPE_FACTOR}" \
    --output_dir ../results_prune/mistral-base-ntk_rope-nodrop/checkpoint

# ─── GQA baseline ──────────────────────────────────────────────────────────────

echo ""
echo "[3/3] GQA baseline (num_kv_heads=${GQA_NUM_KV_HEADS})..."
python scripts/create_variant_baseline.py \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --attention_variant  gqa \
    --gqa_num_kv_heads   "${GQA_NUM_KV_HEADS}" \
    --output_dir ../results_prune/mistral-base-gqa-nodrop/checkpoint

# ─── StreamLLM inference benchmark (nodrop baseline) ───────────────────────────
# Run directly on the base model — avoids loading the checkpoint's custom
# modeling (which applies the calibration-style mask patch) before enable_streaming_llm.
# This gives the clean nodrop latency/memory baseline to compare against pruned runs.

echo ""
echo "[4/4] StreamLLM inference benchmark (nodrop baseline)..."
mkdir -p ../results_prune/mistral-base-streamllm-nodrop
python scripts/benchmark_inference.py \
    --model_path      "${MODEL_NAME_OR_PATH}" \
    --streaming \
    --start_size      "${STREAMLLM_N_INIT}" \
    --recent_size     "${STREAMLLM_N_LOCAL}" \
    --prompt_len      "${PROMPT_LEN}" \
    --gen_len         "${GEN_LEN}" \
    --runs            "${BENCHMARK_RUNS}" \
    2>&1 | tee ../results_prune/mistral-base-streamllm-nodrop/benchmark_streaming.txt

# ─── DONE ──────────────────────────────────────────────────────────────────────

echo ""
echo "========================================"
echo "  Done. Checkpoints:"
echo "    ../results_prune/mistral-base-streamllm-nodrop/checkpoint"
echo "    ../results_prune/mistral-base-ntk_rope-nodrop/checkpoint"
echo "    ../results_prune/mistral-base-gqa-nodrop/checkpoint"
echo "  Benchmark:"
echo "    ../results_prune/mistral-base-streamllm-nodrop/benchmark_streaming.txt"
echo "========================================"
