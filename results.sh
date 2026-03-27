rsync -av --progress \
  --exclude='checkpoint' \
  --exclude='cache' \
  015902406@coe-hpc1.sjsu.edu:~/results_prune/ \
  /Users/ashleyirawan/Desktop/thesisv2/forks/LLM-Drop-v2/results_prune/