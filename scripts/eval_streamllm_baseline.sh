#!/usr/bin/env bash
#SBATCH --job-name=llmdrop-streamllm-baseline
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

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ─── ARGS ──────────────────────────────────────────────────────────────────────
# Usage: sbatch eval_streamllm_baseline.sh <task> <num_fewshot>
# e.g.:  sbatch eval_streamllm_baseline.sh hellaswag 10
#        sbatch eval_streamllm_baseline.sh mmlu 5
#        sbatch eval_streamllm_baseline.sh winogrande 5
#        sbatch eval_streamllm_baseline.sh openbookqa 0

TASK="${1:?Usage: sbatch eval_streamllm_baseline.sh <task> <num_fewshot>}"
FEWSHOT="${2:?Usage: sbatch eval_streamllm_baseline.sh <task> <num_fewshot>}"

# ─── CONFIG ────────────────────────────────────────────────────────────────────

PORT="21304"

CHECKPOINT_DIR="../results_prune/mistral-base-streamllm-nodrop/checkpoint"
EVAL_OUTPUT_DIR="../results_prune/mistral-base-streamllm-nodrop/eval"

# ─── SETUP ─────────────────────────────────────────────────────────────────────

cd ~/LLM-Drop-v2
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "${EVAL_OUTPUT_DIR}" logs

echo "========================================"
echo "  Job        : ${SLURM_JOB_ID}"
echo "  Node       : ${SLURM_NODELIST}"
echo "  Task       : ${TASK} (${FEWSHOT}-shot)"
echo "  Checkpoint : ${CHECKPOINT_DIR}"
echo "  Eval output: ${EVAL_OUTPUT_DIR}"
echo "========================================"

# ─── lm_eval ───────────────────────────────────────────────────────────────────

RESULT_FILE="${EVAL_OUTPUT_DIR}/${FEWSHOT}shot_${TASK}.json"
PERF_LOG="${EVAL_OUTPUT_DIR}/perf_${TASK}.log"

nvidia-smi --query-gpu=memory.used,memory.free,memory.total \
    --format=csv,noheader,nounits --loop=1 \
    >> "${PERF_LOG}" 2>&1 &
NVSMI_PID=$!

TIME_START=$(date +%s)

accelerate launch --num_processes 1 --main_process_port $PORT \
    -m lm_eval \
    --model hf \
    --model_args "pretrained=${CHECKPOINT_DIR},trust_remote_code=True,dtype=float16" \
    --tasks "${TASK}" \
    --num_fewshot "${FEWSHOT}" \
    --batch_size 1 \
    --output_path "${RESULT_FILE}" \
    >> "${EVAL_OUTPUT_DIR}/eval_${TASK}.out" 2>&1

TIME_END=$(date +%s)
kill "${NVSMI_PID}" 2>/dev/null || true

ELAPSED=$((TIME_END - TIME_START))
PEAK_MEM=$(awk -F',' '{gsub(/ /, "", $1); print $1+0}' "${PERF_LOG}" | sort -n | tail -1)

echo "  saved  -> ${RESULT_FILE}"
echo "  time   -> ${ELAPSED}s"
echo "  peak   -> ${PEAK_MEM} MiB GPU memory"
echo "========================================"
