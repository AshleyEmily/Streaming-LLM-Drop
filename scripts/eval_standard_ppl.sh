#!/usr/bin/env bash
#SBATCH --job-name=llmdrop-ppl-std
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

MODEL_PATH="meta-llama/Meta-Llama-3-8B"
NUM_EVAL_TOKENS=400000   # number of tokens to score

# ─── DERIVED PATHS ─────────────────────────────────────────────────────────────

cd ~/LLM-Drop-v2
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"

OUTPUT_DIR="../results_prune/llama3-8b-reference/ppl_native_${SLURM_JOB_ID}"
mkdir -p "${OUTPUT_DIR}" logs

echo "========================================"
echo "  Job         : ${SLURM_JOB_ID}"
echo "  Node        : ${SLURM_NODELIST}"
echo "  Model       : ${MODEL_PATH}"
echo "  eval_tokens : ${NUM_EVAL_TOKENS}"
echo "========================================"

python scripts/eval_standard_ppl.py \
    --model_path      "${MODEL_PATH}" \
    --num_eval_tokens "${NUM_EVAL_TOKENS}" \
    --output_dir      "${OUTPUT_DIR}"

echo ""
echo "  Results: ${OUTPUT_DIR}/metrics.txt"
echo "========================================"
