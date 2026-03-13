#!/usr/bin/env bash
set -euo pipefail

cd /workspace/starVLA

################################################################################
# User config
################################################################################
Framework_name="QwenOFT"
freeze_module_list=""
base_vlm="/mnt/synology/PretrainedModel/Qwen2.5-VL-3B-Instruct-Action"
config_yaml="./examples/LIBERO/train_files/starvla_cotrain_libero.yaml"
libero_data_root="/mnt/synology/RobotData/LEROBOT_LIBERO_DATA"
data_mix="libero_all"
run_root_dir="./results/Checkpoints"
run_id="multinode_libero_qwen25oft"

# training options
per_device_batch_size=24
max_train_steps=10000000
save_interval=10000
logging_frequency=100
eval_interval=100

# wandb
wandb_project="starVLA_Libero"
wandb_entity="jinhuiye"
################################################################################

: "${MASTER_ADDR:?Need MASTER_ADDR}"
: "${MASTER_PORT:?Need MASTER_PORT}"
: "${NODE_RANK:?Need NODE_RANK}"
: "${NNODES:?Need NNODES}"

# Optional overrides
export WANDB_MODE="${WANDB_MODE:-disabled}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1

# Node-local sanity prints
echo "============================================================"
echo "[starVLA multinode launch]"
echo "HOSTNAME              = $(hostname)"
echo "MASTER_ADDR           = ${MASTER_ADDR}"
echo "MASTER_PORT           = ${MASTER_PORT}"
echo "NODE_RANK             = ${NODE_RANK}"
echo "NNODES                = ${NNODES}"
echo "NCCL_SOCKET_IFNAME    = ${NCCL_SOCKET_IFNAME:-<unset>}"
echo "CUDA_VISIBLE_DEVICES  = ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "============================================================"

python -c "import torch; print('torch=', torch.__version__)"
python -c "import torch; print('cuda_available=', torch.cuda.is_available(), 'device_count=', torch.cuda.device_count())"
python -c "import torch; print('nccl=', torch.cuda.nccl.version())"

output_dir="${run_root_dir}/${run_id}"
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"

# NOTE:
# - You have 2 nodes x 1 GPU each
# - total processes = 2
# - machine rank = NODE_RANK (0 on node1, 1 on node2)
#
# We pass --num_machines=2 and --num_processes=2 to match the current repo style.

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_machines "${NNODES}" \
  --num_processes "${NNODES}" \
  --machine_rank "${NODE_RANK}" \
  --main_process_ip "${MASTER_ADDR}" \
  --main_process_port "${MASTER_PORT}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${config_yaml}" \
  --framework.name "${Framework_name}" \
  --framework.qwenvl.base_vlm "${base_vlm}" \
  --datasets.vla_data.data_root_dir "${libero_data_root}" \
  --datasets.vla_data.data_mix "${data_mix}" \
  --datasets.vla_data.per_device_batch_size "${per_device_batch_size}" \
  --trainer.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules "${freeze_module_list}" \
  --trainer.max_train_steps "${max_train_steps}" \
  --trainer.save_interval "${save_interval}" \
  --trainer.logging_frequency "${logging_frequency}" \
  --trainer.eval_interval "${eval_interval}" \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}" \
  --wandb_project "${wandb_project}" \
  --wandb_entity "${wandb_entity}"