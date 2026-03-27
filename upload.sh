# Exclude the large directories
rsync -av --progress \
  --exclude='results/' \
  --exclude='results_cluster/' \
  --exclude='logs_cluster/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='.git' \
  --exclude='results_prune' \
  --exclude='test_streamllm.py' \
  --exclude='test_equivalence.py' \
  --exclude='logs.sh' \
  --exclude='results.sh' \
  --exclude='upload.sh' \
  /Users/ashleyirawan/Desktop/thesisv2/forks/LLM-Drop-v2 \
  015902406@coe-hpc1.sjsu.edu:~/
