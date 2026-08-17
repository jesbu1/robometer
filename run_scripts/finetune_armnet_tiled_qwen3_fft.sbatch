#!/bin/bash
#SBATCH -J rbm_fft_armnet_tiled
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=800G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/ft_armnet_tiled_qwen3_%j.out
#SBATCH --error=slurm_logs/ft_armnet_tiled_qwen3_%j.err
#SBATCH --exclude=g002
#SBATCH --account=xpai
source ~/.bashrc
set -e
ulimit -n 65536
cd /gpfs/scrubbed/jessezha/robometer_jesse
ARMNET_TILED="[jesbu1_armnetbench_v01_tiled_so101_armnetbench_v01_tiled_so101,jesbu1_armnetbench_v01_tiled_bimanual_so101_armnetbench_v01_tiled_bimanual_so101]"
uv run accelerate launch --config_file robometer/configs/distributed/ddp.yaml \
  --num_processes=4 \
  --fsdp_transformer_layer_cls_to_wrap="Qwen3VLVisionBlock,Qwen3VLTextDecoderLayer" \
  train.py \
  model.base_model_id=Qwen/Qwen3-VL-4B-Instruct \
  model.use_peft=false \
  model.train_progress_head=true \
  model.train_preference_head=true \
  data.train_datasets=${ARMNET_TILED} \
  data.eval_datasets=${ARMNET_TILED} \
  data.max_frames=8 \
  data.resized_width=640 \
  data.resized_height=540 \
  data.use_multi_image=true \
  data.dataset_success_cutoff_file=robometer/data/dataset_success_cutoff.txt \
  training.load_from_checkpoint=robometer/Robometer-4B \
  training.exp_name="rbm_fft_armnet_tiled_qwen3_4b_16bs_4gpu" \
  training.per_device_train_batch_size=16 \
  training.per_device_eval_batch_size=16 \
  training.overwrite_output_dir=true \
  training.learning_rate=2e-5 \
  training.max_steps=1000 \
  custom_eval.eval_types=[reward_alignment,policy_ranking] \
  custom_eval.reward_alignment=${ARMNET_TILED} \
  custom_eval.policy_ranking=${ARMNET_TILED} \
  logging.log_to=[] \
  logging.save_best.metric_names=[eval_rew_align/pearson_armnetbench_tiled_so101,eval_p_rank/kendall_last_armnetbench_tiled_so101,eval_rew_align/pearson_armnetbench_tiled_bimanual_so101,eval_p_rank/kendall_last_armnetbench_tiled_bimanual_so101] \
  logging.save_best.greater_is_better=[True,True,True,True]
