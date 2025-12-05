#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

# ckpt_strats=("Blockwise" "Attention" "FFN")
# model_names=("large/48layers" "large/52layers" "large/60layers" "xlarge" "xlarge/52layers" "xlarge/56layers" "xlarge/60layers")
# model_names=("xlarge/68layers" "xlarge/72layers" "xlarge/76layers" "xlarge/80layers")
# ckpt_strats=("FFN")
# model_names=("xlarge/64layers" "xlarge/66layers")
# model_names=("xlarge/67layers")
# ckpt_strats=("Blockwise" "Attention")
# model_names=("xlarge/69layers" "xlarge/70layers" "xlarge/71layers")
ckpt_strats=("Blockwise")
model_names=("large/42layers")
result_file="mem_allopts.csv"
log_file=../results/barbell/mem_allopts.log
# rm $log_file

for ckpt_strat in "${ckpt_strats[@]}"; do
  for model_name in "${model_names[@]}"; do
    echo "Running benchmark for model: $model_name with checkpointing strategy: $ckpt_strat"
    python -m memopt.benchmark.mem_allopts --model_name "$model_name" \
        --ckpt_strat "$ckpt_strat" \
        --result_csv "$result_file" >> "$log_file" 2>&1
  done
done
