#!/usr/bin/env bash
#SBATCH --job-name=llmdrop-debug
#SBATCH --partition=gpuqs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:00:00
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

cd ~/LLM-Drop-v2
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"

PRUNED_MODEL="../results_prune/mistral-base-layer_drop_attn-discrete-drop8-streamllm/checkpoint"

echo "========================================"
echo "  [1/2] Full model"
echo "========================================"
python scripts/debug_ppl.py \
    --model_path mistralai/Mistral-7B-v0.1 \
    --start_size 1 \
    --recent_size 255

echo ""
echo "========================================"
echo "  [2/2] Pruned streamllm model"
echo "========================================"
python scripts/debug_ppl.py \
    --model_path "${PRUNED_MODEL}" \
    --start_size 1 \
    --recent_size 255
