#!/bin/bash
#SBATCH -J busybox_tiled_eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=slurm_logs/busybox_tiled_eval_%j.out
#SBATCH --error=slurm_logs/busybox_tiled_eval_%j.err
#SBATCH --account=weirdlab

source ~/.bashrc
set -e
ulimit -n 65536
cd /gpfs/scrubbed/jessezha/robometer_jesse

CKPT=${1:?checkpoint dir required}
BUSYBOX_TILED="[jesbu1_so101_busybox_push_green_button_tiled_so101_busybox_push_green_button_lerobot_tiled]"

echo "===== Reward alignment (frame steps, 200 trajs) ====="
uv run python robometer/evals/run_baseline_eval.py \
  reward_model=rbm "model_path='${CKPT}'" \
  custom_eval.eval_types=[reward_alignment] \
  custom_eval.reward_alignment=${BUSYBOX_TILED} \
  custom_eval.use_frame_steps=true \
  custom_eval.subsample_n_frames=5 \
  custom_eval.reward_alignment_max_trajectories=200 \
  max_frames=8 model_config.batch_size=32

echo "===== Policy ranking (whole trajectory, 200 per group) ====="
uv run python robometer/evals/run_baseline_eval.py \
  reward_model=rbm "model_path='${CKPT}'" \
  custom_eval.eval_types=[policy_ranking] \
  custom_eval.policy_ranking=${BUSYBOX_TILED} \
  custom_eval.use_frame_steps=false \
  custom_eval.num_examples_per_quality_pr=200 \
  max_frames=8 model_config.batch_size=32

echo "===== DONE $(date) ====="
