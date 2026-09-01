#!/bin/bash
#SBATCH -J molmoact2_tiled_eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=slurm_logs/molmoact2_tiled_eval_%j.out
#SBATCH --error=slurm_logs/molmoact2_tiled_eval_%j.err
#SBATCH --account=xpai
source ~/.bashrc
set -e
ulimit -n 65536
cd /gpfs/scrubbed/jessezha/robometer_jesse

CKPT=/gpfs/scrubbed/jessezha/robometer_jesse/logs/rbm_fft_molmoact2_tiled_qwen3_4b_16bs_4gpu/ckpt-avg-5metrics=0.0000_step=1000
MOLMOACT2_TILED="[jesbu1_molmoact2_yam_tiled_rfm_molmoact2_yam_tiled]"

uv run python robometer/evals/run_baseline_eval.py \
  reward_model=rbm "model_path='${CKPT}'" \
  custom_eval.eval_types=[reward_alignment] \
  custom_eval.reward_alignment=${MOLMOACT2_TILED} \
  custom_eval.use_frame_steps=true \
  custom_eval.subsample_n_frames=5 \
  custom_eval.reward_alignment_max_trajectories=200 \
  max_frames=8 model_config.batch_size=32

uv run python robometer/evals/run_baseline_eval.py \
  reward_model=rbm "model_path='${CKPT}'" \
  custom_eval.eval_types=[policy_ranking] \
  custom_eval.policy_ranking=${MOLMOACT2_TILED} \
  custom_eval.use_frame_steps=false \
  custom_eval.num_examples_per_quality_pr=200 \
  max_frames=8 model_config.batch_size=32
