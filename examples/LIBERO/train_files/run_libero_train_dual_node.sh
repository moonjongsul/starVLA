#!/bin/bash

MASTER_ADDR=192.168.200.12
MASTER_PORT=29500

TRAIN_ARGS="
cd /workspace/starVLA &&
export WANDB_MODE=disabled &&
export PYTHONWARNINGS=ignore::UserWarning:torchvision &&
export NCCL_SOCKET_IFNAME=enp1s0f1np1 &&
export NCCL_IB_DISABLE=0 &&
export NCCL_NET_GDR_LEVEL=5 &&
export NCCL_DEBUG=WARN &&
export TORCH_NCCL_BLOCKING_WAIT=1 &&
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 &&
export NCCL_TIMEOUT=3600 &&
export TORCH_CUDA_ARCH_LIST=12.1+PTX &&
export TORCHINDUCTOR_DISABLE=1 &&
python -m torch.distributed.run \
  --nproc_per_node=1 \
  --nnodes=2 \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  starVLA/training/train_starvla.py \
  --config_yaml ./examples/LIBERO/train_files/starvla_cotrain_libero.yaml \
  --framework.name QwenOFT \
  --framework.qwenvl.base_vlm /mnt/synology/PretrainedModel/Qwen2.5-VL-3B-Instruct-Action \
  --datasets.vla_data.data_root_dir /mnt/synology/RobotData/LEROBOT_LIBERO_DATA \
  --datasets.vla_data.data_mix libero_all \
  --datasets.vla_data.per_device_batch_size 4 \
  --trainer.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules '' \
  --trainer.max_train_steps 80000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 100 \
  --run_root_dir ./results/Checkpoints \
  --run_id 1229_libero4in1_qwen25oft
"

# 노드2 컨테이너에서 worker 실행 (백그라운드)
echo "[INFO] Starting worker on node2 (starVLA-node2)..."
ssh js_spark@192.168.200.13 "docker exec starVLA-node2 bash -c '${TRAIN_ARGS} --node_rank=1'" &
SSH_PID=$!

# 노드1 컨테이너에서 master 실행
echo "[INFO] Starting master on node1 (starVLA-node1)..."
docker exec starVLA-node1 bash -c "${TRAIN_ARGS} --node_rank=0"

wait $SSH_PID
echo "[INFO] Training finished."