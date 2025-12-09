#! /bin/bash
LOG_DIR=../results/barbell/pareto/logs
rm -rf $LOG_DIR
mkdir -p $LOG_DIR

export CUDA_VISIBLE_DEVICES=1

model_config_names=("small" "large")
run_configs=(
  "--runner_cls BaselineRunner"
#   "--runner_cls ActckptRunner --ckpt_strategy Blockwise"
#   "--runner_cls ActckptRunner --ckpt_strategy Attention"
  "--runner_cls ActckptRunner --ckpt_strategy FFN"
  "--runner_cls ZeroOffloadRunner --ckpt_strategy None"
#   "--runner_cls ZeroOffloadRunner --ckpt_strategy Blockwise"
#   "--runner_cls ZeroOffloadRunner --ckpt_strategy Attention"
  "--runner_cls ZeroOffloadRunner --ckpt_strategy FFN"
  "--runner_cls OffloadRunner --ckpt_strategy None"
#   "--runner_cls OffloadRunner --ckpt_strategy Blockwise"
#   "--runner_cls OffloadRunner --ckpt_strategy Attention"
  "--runner_cls OffloadRunner --ckpt_strategy FFN"
  "--runner_cls MixprecRunner --ckpt_strategy None"
#   "--runner_cls MixprecRunner --ckpt_strategy Blockwise"
#   "--runner_cls MixprecRunner --ckpt_strategy Attention"
  "--runner_cls MixprecRunner --ckpt_strategy FFN"
)

for model_config_name in "${model_config_names[@]}"; do
  for run_config in "${run_configs[@]}"; do
    echo "Running benchmark for model config: $model_config_name with config: $run_config"
    python -m memopt.benchmark.benchmark_pareto --model_config_name "$model_config_name" \
      $run_config >> $LOG_DIR/$model_config_name.log 2>&1
  done
done