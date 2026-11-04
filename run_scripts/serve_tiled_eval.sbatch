#!/bin/bash
#SBATCH -J rbm_tiled_eval_server
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_logs/tiled_eval_server_%j.out
#SBATCH --error=slurm_logs/tiled_eval_server_%j.err
#SBATCH --exclude=g002
#SBATCH --account=weirdlab

# Serve a tiled Robometer checkpoint over HTTP. Clients POST raw view videos and
# the server decodes + tiles them before scoring.
#
#   sbatch run_scripts/serve_tiled_eval.sbatch [MODEL_PATH] [PORT]
#
# MODEL_PATH defaults to the busybox tiled checkpoint; PORT defaults to 8010.
# The job prints its node and port so you can tunnel or curl it:
#   ssh -N -L 8010:$(hostname):8010 <user>@<cluster>
#   python scripts/send_videos_to_tiled_eval.py --url http://localhost:8010 ...

set -e
source ~/.bashrc
ulimit -n 65536
cd /gpfs/scrubbed/jessezha/robometer_jesse

MODEL_PATH=${1:-/gpfs/scrubbed/jessezha/robometer_jesse/logs/rbm_fft_busybox_tiled_qwen3_4b_16bs_4gpu/ckpt-avg-2metrics=0.9219_step=750}
PORT=${2:-8010}

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Model: ${MODEL_PATH}"
echo "URL for clients: http://$(hostname):${PORT}/evaluate_videos"
echo "Start Time: $(date)"

uv run python robometer/evals/eval_server.py \
  model_path="'${MODEL_PATH}'" \
  num_gpus=1 \
  batch_size=4 \
  server_url=0.0.0.0 \
  server_port="${PORT}"

echo "End Time: $(date)"
