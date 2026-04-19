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

STREAMLLM_N_INIT=4
STREAMLLM_N_LOCAL=4092
NTK_ROPE_FACTOR=4.0
GQA_NUM_KV_HEADS=4

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
echo "[1/4] StreamLLM baseline (n_init=${STREAMLLM_N_INIT}, n_local=${STREAMLLM_N_LOCAL})..."
python scripts/create_variant_baseline.py \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --attention_variant  streamllm \
    --streamllm_n_init   "${STREAMLLM_N_INIT}" \
    --streamllm_n_local  "${STREAMLLM_N_LOCAL}" \
    --output_dir ../results_prune/mistral-base-streamllm-nodrop/checkpoint

# ─── NTK-RoPE baseline ─────────────────────────────────────────────────────────

echo ""
echo "[2/4] NTK-RoPE baseline (factor=${NTK_ROPE_FACTOR})..."
python scripts/create_variant_baseline.py \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --attention_variant  ntk_rope \
    --ntk_rope_factor    "${NTK_ROPE_FACTOR}" \
    --output_dir ../results_prune/mistral-base-ntk_rope-nodrop/checkpoint

# ─── GQA baseline ──────────────────────────────────────────────────────────────

echo ""
echo "[3/4] GQA baseline (num_kv_heads=${GQA_NUM_KV_HEADS})..."
python scripts/create_variant_baseline.py \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --attention_variant  gqa \
    --gqa_num_kv_heads   "${GQA_NUM_KV_HEADS}" \
    --output_dir ../results_prune/mistral-base-gqa-nodrop/checkpoint

# ─── Llama 3 StreamLLM baseline ────────────────────────────────────────────────

LLAMA_MODEL="meta-llama/Meta-Llama-3-8B"
LLAMA_STREAMLLM_N_INIT=4
LLAMA_STREAMLLM_N_LOCAL=8188

echo ""
echo "[4/4] Llama 3 StreamLLM baseline (n_init=${LLAMA_STREAMLLM_N_INIT}, n_local=${LLAMA_STREAMLLM_N_LOCAL})..."
python scripts/create_variant_baseline.py \
    --model_name_or_path "${LLAMA_MODEL}" \
    --attention_variant  streamllm \
    --streamllm_n_init   "${LLAMA_STREAMLLM_N_INIT}" \
    --streamllm_n_local  "${LLAMA_STREAMLLM_N_LOCAL}" \
    --output_dir ../results_prune/llama3-8b-streamllm-nodrop/checkpoint

echo ""
echo "========================================"
echo "  Done. Checkpoints:"
echo "    ../results_prune/mistral-base-streamllm-nodrop/checkpoint"
echo "    ../results_prune/mistral-base-ntk_rope-nodrop/checkpoint"
echo "    ../results_prune/mistral-base-gqa-nodrop/checkpoint"
echo "    ../results_prune/llama3-8b-streamllm-nodrop/checkpoint"
echo "========================================"
