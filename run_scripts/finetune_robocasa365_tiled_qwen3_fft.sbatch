#!/bin/bash
#SBATCH -J rbm_fft_robocasa365_tiled
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=800G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/ft_robocasa365_tiled_qwen3_%j.out
#SBATCH --error=slurm_logs/ft_robocasa365_tiled_qwen3_%j.err
#SBATCH --exclude=g002
#SBATCH --account=xpai
source ~/.bashrc
set -e
ulimit -n 65536
cd /gpfs/scrubbed/jessezha/robometer_jesse
ROBOCASA365_TILED="[jesbu1_robocasa365_tiled_rfm_robocasa365_tiled]"
uv run accelerate launch --config_file robometer/configs/distributed/ddp.yaml \
  --num_processes=4 \
  --fsdp_transformer_layer_cls_to_wrap="Qwen3VLVisionBlock,Qwen3VLTextDecoderLayer" \
  train.py \
  model.base_model_id=Qwen/Qwen3-VL-4B-Instruct \
  model.use_peft=false \
  model.train_progress_head=true \
  model.train_preference_head=true \
  data.train_datasets=${ROBOCASA365_TILED} \
  data.eval_datasets=${ROBOCASA365_TILED} \
  data.max_frames=8 \
  data.resized_width=512 \
  data.resized_height=768 \
  data.use_multi_image=true \
  data.dataset_success_cutoff_file=robometer/data/dataset_success_cutoff.txt \
  training.load_from_checkpoint=robometer/Robometer-4B \
  training.exp_name="rbm_fft_robocasa365_tiled_qwen3_4b_16bs_4gpu" \
  training.per_device_train_batch_size=16 \
  training.per_device_eval_batch_size=16 \
  training.overwrite_output_dir=true \
  training.learning_rate=2e-5 \
  training.max_steps=1000 \
  custom_eval.eval_types=[reward_alignment,policy_ranking] \
  custom_eval.reward_alignment=${ROBOCASA365_TILED} \
  custom_eval.policy_ranking=${ROBOCASA365_TILED} \
  logging.log_to=[] \
  logging.save_best.metric_names=[eval_rew_align/pearson_robocasa365_tiled,eval_p_rank/kendall_last_robocasa365_tiled] \
  logging.save_best.greater_is_better=[True,True]