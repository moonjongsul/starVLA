#!/bin/bash

MASTER_ADDR=192.168.200.12
MASTER_PORT=29500

# 노드별 실행 스크립트 생성
cat > /tmp/train_node.sh << 'SCRIPT'
#!/bin/bash
NODE_RANK=$1

export WANDB_MODE=disabled
export PYTHONWARNINGS=ignore::UserWarning:torchvision
export NCCL_SOCKET_IFNAME=enp1s0f1np1
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
export NCCL_DEBUG=WARN
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=3600
export TORCH_CUDA_ARCH_LIST=12.1+PTX
export TORCHINDUCTOR_DISABLE=1

cd /workspace/starVLA

python -m torch.distributed.run \
  --nproc_per_node=1 \
  --nnodes=2 \
  --node_rank=${NODE_RANK} \
  --master_addr=192.168.200.12 \
  --master_port=29500 \
  starVLA/training/train_starvla.py \
  --config_yaml ./examples/LIBERO/train_files/starvla_cotrain_libero.yaml \
  --framework.name QwenOFT \
  --framework.qwenvl.base_vlm /mnt/synology/PretrainedModel/Qwen2.5-VL-3B-Instruct-Action \
  --datasets.vla_data.data_root_dir /mnt/synology/RobotData/LEROBOT_LIBERO_DATA \
  --datasets.vla_data.data_mix libero_all \
  --datasets.vla_data.per_device_batch_size 12 \
  --trainer.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules '' \
  --trainer.max_train_steps 80000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 100 \
  --trainer.dtype bf16 \
  --trainer.is_resume True \
  --run_root_dir ./results/Checkpoints \
  --run_id 1229_libero4in1_qwen25oft_bf16
  # --is_debug True
SCRIPT

chmod +x /tmp/train_node.sh

# 노드2에 스크립트 복사
scp /tmp/train_node.sh js_spark@192.168.200.13:/tmp/train_node.sh

# 노드2 worker 실행 (백그라운드)
echo "[INFO] Starting worker on node2 (starVLA-node2)..."
ssh js_spark@192.168.200.13 "docker cp /tmp/train_node.sh starVLA-node2:/tmp/train_node.sh && docker exec starVLA-node2 bash /tmp/train_node.sh 1" 2>&1 | sed 's/^/[node2] /' &
SSH_PID=$!

sleep 3

# 노드1 master 실행
echo "[INFO] Starting master on node1 (starVLA-node1)..."
docker cp /tmp/train_node.sh starVLA-node1:/tmp/train_node.sh
docker exec starVLA-node1 bash /tmp/train_node.sh 0 2>&1 | sed 's/^/[node1] /'

wait $SSH_PID
echo "[INFO] Training finished."